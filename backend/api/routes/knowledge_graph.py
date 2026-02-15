"""
AIDocumentIndexer - Knowledge Graph API Routes
===============================================

API endpoints for knowledge graph visualization and exploration.
Provides access to entities, relationships, and graph statistics.
"""

import uuid
from datetime import datetime, timezone
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
    EntityType, RelationType, Document, AccessTier,
)
from backend.api.middleware.auth import get_org_filter, AuthenticatedUser, get_user_uuid, safe_uuid
from backend.services.knowledge_graph import get_knowledge_graph_service

logger = structlog.get_logger(__name__)


# =============================================================================
# Access Tier Filtering Helper
# =============================================================================

async def get_accessible_entity_ids(
    db: AsyncSession,
    user: AuthenticatedUser,
    org_id: Optional[uuid.UUID],
) -> set:
    """
    Get entity IDs accessible to user based on document access tiers.

    The knowledge graph extracts entities from ALL documents.
    This function filters which entities a specific user can see
    based on their access tier and private document permissions.

    An entity is accessible if it has at least one mention in a
    document the user can access.
    """
    query = (
        select(EntityMention.entity_id)
        .distinct()
        .join(Document, EntityMention.document_id == Document.id)
        .join(AccessTier, Document.access_tier_id == AccessTier.id)
        .where(AccessTier.level <= user.access_tier_level)
    )

    # Filter private documents - only owner or superadmin can access
    if not user.is_superadmin:
        query = query.where(
            or_(
                Document.is_private == False,
                and_(
                    Document.is_private == True,
                    Document.uploaded_by_id == get_user_uuid(user),
                ),
            )
        )

    # Apply organization filtering
    if org_id and not user.is_superadmin:
        query = query.where(
            or_(
                Document.organization_id == org_id,
                Document.organization_id.is_(None),
            )
        )

    result = await db.execute(query)
    return {row[0] for row in result.fetchall()}

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

    Note: Results are filtered based on user's access tier level.
    Users only see stats for entities from documents they can access.
    """
    try:
        # PHASE 12 FIX: Apply organization filtering for multi-tenant isolation
        org_id = get_org_filter(user)

        # SECURITY FIX: Get entities accessible based on document access tiers
        # Users only see entities from documents at or below their tier level
        accessible_entity_ids = await get_accessible_entity_ids(db, user, org_id)

        # If user has no accessible entities, return empty stats
        if not accessible_entity_ids:
            return GraphStatsResponse(
                total_entities=0,
                total_relations=0,
                total_mentions=0,
                entity_type_distribution={},
                relation_type_distribution={},
                top_entities=[],
                documents_with_entities=0,
            )

        # Build base queries filtered by accessible entities
        entity_base = select(func.count(Entity.id)).where(Entity.id.in_(accessible_entity_ids))
        relation_base = select(func.count(EntityRelation.id)).where(
            and_(
                EntityRelation.source_entity_id.in_(accessible_entity_ids),
                EntityRelation.target_entity_id.in_(accessible_entity_ids),
            )
        )
        mention_base = select(func.count(EntityMention.id)).where(
            EntityMention.entity_id.in_(accessible_entity_ids)
        )

        # Get basic counts
        entity_count = await db.scalar(entity_base) or 0
        relation_count = await db.scalar(relation_base) or 0
        mention_count = await db.scalar(mention_base) or 0

        # Entity type distribution - filtered by accessible entities
        entity_type_query = (
            select(Entity.entity_type, func.count(Entity.id))
            .where(Entity.id.in_(accessible_entity_ids))
            .group_by(Entity.entity_type)
        )
        result = await db.execute(entity_type_query)
        entity_type_dist = {str(row[0].value): row[1] for row in result.all()}

        # Relation type distribution - only relations between accessible entities
        relation_type_query = (
            select(EntityRelation.relation_type, func.count(EntityRelation.id))
            .where(
                and_(
                    EntityRelation.source_entity_id.in_(accessible_entity_ids),
                    EntityRelation.target_entity_id.in_(accessible_entity_ids),
                )
            )
            .group_by(EntityRelation.relation_type)
        )
        result = await db.execute(relation_type_query)
        relation_type_dist = {str(row[0].value): row[1] for row in result.all()}

        # Top entities by mention count - filtered by accessible entities
        mention_counts = (
            select(
                EntityMention.entity_id,
                func.count(EntityMention.id).label("mention_count")
            )
            .where(EntityMention.entity_id.in_(accessible_entity_ids))
            .group_by(EntityMention.entity_id)
            .subquery()
        )

        top_entities_query = (
            select(Entity, mention_counts.c.mention_count)
            .outerjoin(mention_counts, Entity.id == mention_counts.c.entity_id)
            .where(Entity.id.in_(accessible_entity_ids))
            .order_by(desc(mention_counts.c.mention_count))
            .limit(10)
        )

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

        # Documents with entities - count only accessible documents
        docs_with_entities_query = (
            select(func.count(func.distinct(EntityMention.document_id)))
            .join(Document, EntityMention.document_id == Document.id)
            .join(AccessTier, Document.access_tier_id == AccessTier.id)
            .where(AccessTier.level <= user.access_tier_level)
        )
        if not user.is_superadmin:
            docs_with_entities_query = docs_with_entities_query.where(
                or_(
                    Document.is_private == False,
                    and_(
                        Document.is_private == True,
                        Document.uploaded_by_id == get_user_uuid(user),
                    ),
                )
            )
        if org_id and not user.is_superadmin:
            docs_with_entities_query = docs_with_entities_query.where(
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
            detail="Failed to get graph statistics"
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

    Note: Results are filtered based on user's access tier level.
    Users only see entities from documents they can access.
    """
    try:
        # Get organization and accessible entities based on access tier
        org_id = get_org_filter(user)

        # SECURITY FIX: Get entities accessible based on document access tiers
        accessible_entity_ids = await get_accessible_entity_ids(db, user, org_id)

        # If user has no accessible entities, return empty graph
        if not accessible_entity_ids:
            return GraphDataResponse(
                nodes=[],
                edges=[],
                stats=GraphStatsResponse(),
            )

        # Build entity query filtered by accessible entities
        entity_query = select(Entity).where(Entity.id.in_(accessible_entity_ids))
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

        # Get relations where BOTH source and target are accessible (security)
        # but at least one end is in the displayed entity_ids
        relations = []
        if entity_ids:
            # Find relations where at least one end is in displayed entities
            # AND both ends are in the accessible set (security filter)
            relations_query = (
                select(EntityRelation)
                .options(
                    selectinload(EntityRelation.source_entity),
                    selectinload(EntityRelation.target_entity),
                )
                .where(
                    and_(
                        # Security: both ends must be accessible
                        EntityRelation.source_entity_id.in_(accessible_entity_ids),
                        EntityRelation.target_entity_id.in_(accessible_entity_ids),
                        # At least one end is in the displayed set
                        or_(
                            EntityRelation.source_entity_id.in_(entity_ids),
                            EntityRelation.target_entity_id.in_(entity_ids),
                        ),
                    )
                )
            )
            result = await db.execute(relations_query)
            relations = list(result.scalars().all())

            # Add missing connected entities to the node list
            connected_entity_ids = set()
            for rel in relations:
                connected_entity_ids.add(rel.source_entity_id)
                connected_entity_ids.add(rel.target_entity_id)

            # Fetch any connected entities not already in our list
            missing_ids = connected_entity_ids - set(entity_ids)
            if missing_ids:
                missing_query = select(Entity).where(Entity.id.in_(missing_ids))
                missing_result = await db.execute(missing_query)
                entities.extend(missing_result.scalars().all())

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
            detail="Failed to get graph data"
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

    Note: Results are filtered based on user's access tier level.
    Users only see entities from documents they can access.
    """
    try:
        # Get organization and accessible entities based on access tier
        org_id = get_org_filter(user)

        # SECURITY FIX: Get entities accessible based on document access tiers
        accessible_entity_ids = await get_accessible_entity_ids(db, user, org_id)

        # If user has no accessible entities, return empty results
        if not accessible_entity_ids:
            return EntitySearchResponse(
                entities=[],
                total=0,
                page=page,
                page_size=page_size,
                has_more=False,
            )

        # Build base query filtered by accessible entities
        base_query = select(Entity).where(Entity.id.in_(accessible_entity_ids))
        count_query = select(func.count(Entity.id)).where(Entity.id.in_(accessible_entity_ids))

        conditions = []

        # Name search
        if query:
            safe_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            search_pattern = f"%{safe_query}%"
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
        base_query = base_query.order_by(Entity.name.asc()).offset(offset).limit(page_size)

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
            detail="Failed to search entities"
        )


@router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get a specific entity by ID.

    Note: User must have access to at least one document mentioning this entity.
    """
    try:
        entity_uuid = uuid.UUID(entity_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid entity ID format"
        )

    # Get organization and accessible entities based on access tier
    org_id = get_org_filter(user)

    # SECURITY FIX: Check if user can access this entity
    accessible_entity_ids = await get_accessible_entity_ids(db, user, org_id)
    if entity_uuid not in accessible_entity_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Entity not found"
        )

    entity_query = select(Entity).where(Entity.id == entity_uuid)
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


class EntityCreate(BaseModel):
    """Request model for creating an entity."""
    name: str = Field(..., min_length=1, max_length=500)
    entity_type: str = Field(..., description="Entity type (e.g., PERSON, ORGANIZATION)")
    description: Optional[str] = None
    aliases: Optional[List[str]] = None


class EntityUpdate(BaseModel):
    """Request model for updating an entity."""
    name: Optional[str] = None
    description: Optional[str] = None
    aliases: Optional[List[str]] = None
    entity_type: Optional[str] = None


class RelationCreate(BaseModel):
    """Request model for creating a relation."""
    source_entity_id: str
    target_entity_id: str
    relation_type: str = Field(..., description="Relation type (e.g., RELATED_TO, WORKS_FOR)")
    relation_label: Optional[str] = None
    description: Optional[str] = None
    weight: float = 1.0


class RelationUpdate(BaseModel):
    """Request model for updating a relation."""
    relation_type: Optional[str] = None
    relation_label: Optional[str] = None
    description: Optional[str] = None
    weight: Optional[float] = None


class DeleteResponse(BaseModel):
    """Response for delete operations."""
    success: bool
    message: str


@router.post("/entities", response_model=EntityResponse, status_code=status.HTTP_201_CREATED)
async def create_entity(
    entity_data: EntityCreate,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Manually create a new entity in the knowledge graph.
    """
    # Validate entity type
    try:
        entity_type_enum = EntityType(entity_data.entity_type.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid entity type: {entity_data.entity_type}. Valid types: {[e.value for e in EntityType]}"
        )

    org_id = get_org_filter(user)

    entity = Entity(
        name=entity_data.name,
        name_normalized=entity_data.name.lower().strip(),
        entity_type=entity_type_enum,
        description=entity_data.description,
        aliases=entity_data.aliases or [],
        organization_id=org_id,
        extraction_method="user_defined",
        confidence=1.0,
    )
    db.add(entity)
    await db.commit()
    await db.refresh(entity)

    logger.info("Entity created manually", entity_id=str(entity.id), name=entity.name)

    return EntityResponse(
        id=str(entity.id),
        name=entity.name,
        entity_type=entity.entity_type.value,
        description=entity.description,
        aliases=entity.aliases or [],
        mention_count=0,
        created_at=entity.created_at.isoformat() if entity.created_at else "",
    )


@router.patch("/entities/{entity_id}", response_model=EntityResponse)
async def update_entity(
    entity_id: str,
    update: EntityUpdate,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update an entity in the knowledge graph.

    Allows updating the entity's name, description, aliases, or type.
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

    # Apply updates
    if update.name is not None:
        entity.name = update.name

    if update.description is not None:
        entity.description = update.description

    if update.aliases is not None:
        entity.aliases = update.aliases

    if update.entity_type is not None:
        try:
            entity.entity_type = EntityType(update.entity_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid entity type: {update.entity_type}. Valid types: {[e.value for e in EntityType]}"
            )

    await db.commit()
    await db.refresh(entity)

    # Get mention count
    mention_count = await db.scalar(
        select(func.count(EntityMention.id))
        .where(EntityMention.entity_id == entity_uuid)
    ) or 0

    logger.info("Entity updated", entity_id=entity_id, updates=update.model_dump(exclude_none=True))

    return EntityResponse(
        id=str(entity.id),
        name=entity.name,
        entity_type=entity.entity_type.value,
        description=entity.description,
        aliases=entity.aliases or [],
        mention_count=mention_count,
        created_at=entity.created_at.isoformat() if entity.created_at else "",
    )


@router.delete("/entities/{entity_id}", response_model=DeleteResponse)
async def delete_entity(
    entity_id: str,
    user: AuthenticatedUser,
    cascade: bool = Query(False, description="Also delete all relations involving this entity"),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete an entity from the knowledge graph.

    By default, this will fail if the entity has relations. Use cascade=true
    to also delete all relations involving this entity.
    """
    from sqlalchemy import delete

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

    # Check for existing relations
    relation_count = await db.scalar(
        select(func.count(EntityRelation.id))
        .where(
            or_(
                EntityRelation.source_entity_id == entity_uuid,
                EntityRelation.target_entity_id == entity_uuid,
            )
        )
    ) or 0

    if relation_count > 0 and not cascade:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Entity has {relation_count} relations. Use cascade=true to delete them as well."
        )

    entity_name = entity.name

    # Delete relations if cascading
    if cascade and relation_count > 0:
        await db.execute(
            delete(EntityRelation)
            .where(
                or_(
                    EntityRelation.source_entity_id == entity_uuid,
                    EntityRelation.target_entity_id == entity_uuid,
                )
            )
        )

    # Delete entity mentions
    await db.execute(
        delete(EntityMention).where(EntityMention.entity_id == entity_uuid)
    )

    # Delete the entity
    await db.delete(entity)
    await db.commit()

    logger.info(
        "Entity deleted",
        entity_id=entity_id,
        entity_name=entity_name,
        cascade=cascade,
        relations_deleted=relation_count if cascade else 0,
    )

    return DeleteResponse(
        success=True,
        message=f"Entity '{entity_name}' deleted successfully" +
                (f" along with {relation_count} relations" if cascade and relation_count > 0 else "")
    )


# =============================================================================
# Relation CRUD Endpoints
# =============================================================================

@router.post("/relations", response_model=RelationResponse, status_code=status.HTTP_201_CREATED)
async def create_relation(
    relation_data: RelationCreate,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Manually create a new relation between two entities.
    """
    # Validate relation type
    try:
        relation_type_enum = RelationType(relation_data.relation_type.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid relation type: {relation_data.relation_type}. Valid types: {[r.value for r in RelationType]}"
        )

    # Validate entity IDs
    try:
        source_uuid = uuid.UUID(relation_data.source_entity_id)
        target_uuid = uuid.UUID(relation_data.target_entity_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid entity ID format"
        )

    # Verify both entities exist
    source = await db.get(Entity, source_uuid)
    target = await db.get(Entity, target_uuid)
    if not source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source entity not found")
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target entity not found")

    org_id = get_org_filter(user)

    # Multi-tenant isolation: ensure both entities belong to user's org
    if org_id:
        if (source.organization_id and source.organization_id != org_id) or \
           (target.organization_id and target.organization_id != org_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entity not found")

    relation = EntityRelation(
        source_entity_id=source_uuid,
        target_entity_id=target_uuid,
        relation_type=relation_type_enum,
        relation_label=relation_data.relation_label,
        description=relation_data.description,
        weight=relation_data.weight,
        organization_id=org_id,
        confidence=1.0,
    )
    db.add(relation)
    await db.commit()
    await db.refresh(relation)

    logger.info("Relation created manually", relation_id=str(relation.id),
                source=source.name, target=target.name)

    return RelationResponse(
        id=str(relation.id),
        source_entity_id=str(source_uuid),
        source_entity_name=source.name,
        target_entity_id=str(target_uuid),
        target_entity_name=target.name,
        relation_type=relation.relation_type.value,
        relation_label=relation.relation_label,
        weight=relation.weight,
    )


@router.patch("/relations/{relation_id}", response_model=RelationResponse)
async def update_relation(
    relation_id: str,
    update: RelationUpdate,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update an existing relation in the knowledge graph.
    """
    try:
        relation_uuid = uuid.UUID(relation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid relation ID format"
        )

    result = await db.execute(
        select(EntityRelation)
        .options(selectinload(EntityRelation.source_entity), selectinload(EntityRelation.target_entity))
        .where(EntityRelation.id == relation_uuid)
    )
    relation = result.scalar_one_or_none()

    if not relation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Relation not found")

    if update.relation_type is not None:
        try:
            relation.relation_type = RelationType(update.relation_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid relation type: {update.relation_type}. Valid types: {[r.value for r in RelationType]}"
            )

    if update.relation_label is not None:
        relation.relation_label = update.relation_label
    if update.description is not None:
        relation.description = update.description
    if update.weight is not None:
        relation.weight = update.weight

    await db.commit()
    await db.refresh(relation)

    logger.info("Relation updated", relation_id=relation_id, updates=update.model_dump(exclude_none=True))

    return RelationResponse(
        id=str(relation.id),
        source_entity_id=str(relation.source_entity_id),
        source_entity_name=relation.source_entity.name,
        target_entity_id=str(relation.target_entity_id),
        target_entity_name=relation.target_entity.name,
        relation_type=relation.relation_type.value,
        relation_label=relation.relation_label,
        weight=relation.weight,
    )


@router.delete("/relations/{relation_id}", response_model=DeleteResponse)
async def delete_relation(
    relation_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete a relation from the knowledge graph.
    """
    try:
        relation_uuid = uuid.UUID(relation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid relation ID format"
        )

    relation = await db.get(EntityRelation, relation_uuid)
    if not relation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Relation not found")

    await db.delete(relation)
    await db.commit()

    logger.info("Relation deleted", relation_id=relation_id)

    return DeleteResponse(success=True, message="Relation deleted successfully")


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

    Note: Results are filtered based on user's access tier level.
    Neighbors are limited to entities the user can access.
    """
    try:
        entity_uuid = uuid.UUID(entity_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid entity ID format"
        )

    # Get organization and accessible entities based on access tier
    org_id = get_org_filter(user)

    # SECURITY FIX: Get accessible entities and verify requested entity is accessible
    accessible_entity_ids = await get_accessible_entity_ids(db, user, org_id)
    if entity_uuid not in accessible_entity_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Entity not found"
        )

    # Get the entity
    entity_query = select(Entity).where(Entity.id == entity_uuid)
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

    # SECURITY FIX: Filter neighbors to only include accessible entities
    accessible_neighbors = [n for n in neighbors if n.id in accessible_entity_ids]

    # SECURITY FIX: Filter relations to only include those between accessible entities
    accessible_relations = [
        r for r in relations
        if r.source_entity_id in accessible_entity_ids
        and r.target_entity_id in accessible_entity_ids
    ]

    # Get mention counts for accessible entities
    entity_ids = [n.id for n in accessible_neighbors]
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
        for n in accessible_neighbors if n.id != entity_uuid
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
        for r in accessible_relations
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
        document_name=document.filename or document.original_filename,
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
            detail="Failed to extract entities"
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
            detail="Failed to extract entities"
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
            detail="Failed to cleanup graph"
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


class ExtractionJobDocumentDetail(BaseModel):
    """Per-document detail within an extraction job."""
    document_id: str
    filename: str
    status: str  # pending, processing, completed, failed
    chunk_count: int = 0
    chunks_processed: Optional[int] = None
    kg_entity_count: int = 0
    kg_relation_count: int = 0


class ExtractionJobDocumentsResponse(BaseModel):
    """Response containing per-document details for an extraction job."""
    job_id: str
    documents: List[ExtractionJobDocumentDetail]


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
        org_id = safe_uuid(user.organization_id)
        existing = await service.get_running_job(org_id)
        if existing:
            return ExtractionJobResponse(
                job_id=str(existing.id),
                status="already_running",
                message="An extraction job is already running. Check its progress or cancel it first.",
            )

        # Create new job
        # When specific document_ids are provided, force only_new_documents=False
        # so the user can re-extract already-completed documents
        effective_only_new = request.only_new_documents
        if request.document_ids:
            effective_only_new = False

        job = await service.create_job(
            user_id=get_user_uuid(user),
            organization_id=org_id,
            only_new_documents=effective_only_new,
            document_ids=request.document_ids,
            provider_id=request.provider_id,
        )

        # Dispatch to Celery worker (runs completely separate from backend)
        # This prevents KG extraction from blocking the API
        from backend.tasks.document_tasks import run_kg_extraction_job
        run_kg_extraction_job.delay(
            job_id=str(job.id),
            user_id=user.user_id,
            organization_id=str(org_id) if org_id else None,
            provider_id=str(request.provider_id) if request.provider_id else None,
        )

        # Update job status to indicate it's queued for Celery
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        await db.commit()

        logger.info(
            "Started KG extraction job via Celery",
            job_id=str(job.id),
            user_id=user.user_id,
            only_new=request.only_new_documents,
        )

        return ExtractionJobResponse(
            job_id=str(job.id),
            status="started",
            message="Extraction job started in background. You can navigate away and check progress later.",
        )

    except ValueError as e:
        logger.warning("Invalid extraction job request", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid extraction job request",
        )
    except Exception as e:
        logger.error("Failed to start extraction job", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start extraction job",
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
        org_id = safe_uuid(user.organization_id)
        job = await service.get_running_job(org_id)

        if not job:
            return None

        progress = await service.get_progress(job.id)
        return ExtractionJobProgressResponse(**progress)

    except Exception as e:
        logger.error("Failed to get current extraction job", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get extraction job",
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
        org_id = safe_uuid(user.organization_id)

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
            detail="Failed to get pending count",
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
            detail="Failed to get job progress",
        )


@router.get("/extraction-jobs/{job_id}/documents", response_model=ExtractionJobDocumentsResponse)
async def get_extraction_job_documents(
    job_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Get per-document detail for an extraction job, including chunk progress and entity counts."""
    from backend.services.kg_extraction_job import KGExtractionJobService

    try:
        service = KGExtractionJobService(db)
        details = await service.get_job_documents_detail(uuid.UUID(job_id))

        if details is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Extraction job not found",
            )

        return ExtractionJobDocumentsResponse(
            job_id=job_id,
            documents=[ExtractionJobDocumentDetail(**d) for d in details],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get extraction job documents", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get job documents",
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
            detail="Failed to cancel job",
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
            detail="Failed to pause job",
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
            detail="Failed to resume job",
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
        org_id = safe_uuid(user.organization_id)

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
            detail="Failed to list jobs",
        )
