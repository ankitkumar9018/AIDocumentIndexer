"""
AIDocumentIndexer - Knowledge Graph Service (GraphRAG)
======================================================

Implements GraphRAG for multi-hop reasoning using a knowledge graph.
Extracts entities and relationships from documents and enables
graph-based retrieval for complex queries.

Features:
- Entity extraction using LLM
- Relationship extraction
- Graph traversal for multi-hop reasoning
- Hybrid retrieval (vector + graph)
- Entity resolution and deduplication
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import structlog
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db.models import (
    Entity, EntityMention, EntityRelation,
    EntityType, RelationType,
    Document, Chunk,
)
from services.llm import get_llm_service
from services.embeddings import get_embedding_service

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractedEntity:
    """Entity extracted from text."""
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    context: Optional[str] = None


@dataclass
class ExtractedRelation:
    """Relationship extracted from text."""
    source_entity: str
    target_entity: str
    relation_type: RelationType
    relation_label: Optional[str] = None
    description: Optional[str] = None
    confidence: float = 1.0


@dataclass
class GraphSearchResult:
    """Result from graph-based search."""
    entity: Entity
    relevance_score: float
    path_length: int  # Hops from query entities
    connected_entities: List[Entity] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)


@dataclass
class GraphRAGContext:
    """Combined context from graph and vector search."""
    entities: List[Entity]
    relations: List[EntityRelation]
    chunks: List[Chunk]
    graph_summary: str
    entity_context: str


# =============================================================================
# Entity Extraction Prompts
# =============================================================================

ENTITY_EXTRACTION_PROMPT = """Extract named entities and their relationships from the following text.

Text:
{text}

Extract entities of these types:
- PERSON: People, individuals
- ORGANIZATION: Companies, institutions, groups
- LOCATION: Places, cities, countries
- CONCEPT: Abstract ideas, methodologies, theories
- EVENT: Occurrences, meetings, incidents
- PRODUCT: Products, services, offerings
- TECHNOLOGY: Technologies, tools, systems
- DATE: Dates, time periods
- METRIC: Numbers, statistics, KPIs
- OTHER: Any other notable entities

For each entity, provide:
1. name: The entity name as it appears
2. type: One of the types above
3. description: Brief description if available from context
4. aliases: Alternative names or abbreviations if any

Also extract relationships between entities:
- WORKS_FOR: Person works for organization
- LOCATED_IN: Entity is located in a place
- RELATED_TO: General relationship
- PART_OF: Entity is part of another
- CREATED_BY: Entity was created by another
- USES: Entity uses another
- MENTIONS: Document mentions entity
- CAUSES: One event causes another
- CONTAINS: Entity contains another
- SIMILAR_TO: Entities are similar

Return JSON in this exact format:
{{
  "entities": [
    {{"name": "...", "type": "PERSON|ORGANIZATION|...", "description": "...", "aliases": []}}
  ],
  "relations": [
    {{"source": "entity_name", "target": "entity_name", "type": "WORKS_FOR|...", "label": "optional label"}}
  ]
}}

Only include entities and relations clearly supported by the text. Be precise and avoid speculation."""


# =============================================================================
# Knowledge Graph Service
# =============================================================================

class KnowledgeGraphService:
    """
    Service for building and querying the knowledge graph.

    Implements GraphRAG for enhanced retrieval through:
    1. Entity extraction from documents
    2. Relationship extraction
    3. Graph-based retrieval
    4. Hybrid search (vector + graph)

    LLM Provider Options:
    - FREE: Set OLLAMA_ENABLED=true and use local Ollama with llama3.2
    - PAID: Set OPENAI_API_KEY or ANTHROPIC_API_KEY

    The service automatically uses the configured LLM provider from
    environment variables or database settings.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        llm_service=None,
        embedding_service=None,
    ):
        self.db = db_session
        self.llm = llm_service
        self.embeddings = embedding_service

    async def _get_llm(self):
        """
        Get or initialize LLM service.

        Uses the configured provider from environment or database.
        Supports FREE (Ollama) and PAID (OpenAI, Anthropic) providers.
        """
        if not self.llm:
            self.llm = await get_llm_service()
        return self.llm

    async def _get_embeddings(self):
        """Get or initialize embedding service."""
        if not self.embeddings:
            self.embeddings = await get_embedding_service()
        return self.embeddings

    # -------------------------------------------------------------------------
    # Entity Extraction
    # -------------------------------------------------------------------------

    async def extract_entities_from_text(
        self,
        text: str,
        document_id: Optional[uuid.UUID] = None,
        chunk_id: Optional[uuid.UUID] = None,
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        Extract entities and relationships from text using LLM.

        Args:
            text: Text to extract from
            document_id: Source document ID
            chunk_id: Source chunk ID

        Returns:
            Tuple of (entities, relations)
        """
        llm = await self._get_llm()

        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:8000])  # Limit text size

        try:
            response = await llm.generate(prompt)

            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                logger.warning("No JSON found in entity extraction response")
                return [], []

            data = json.loads(json_match.group())

            entities = []
            for e in data.get("entities", []):
                try:
                    entity_type = EntityType(e.get("type", "other").lower())
                except ValueError:
                    entity_type = EntityType.OTHER

                entities.append(ExtractedEntity(
                    name=e.get("name", ""),
                    entity_type=entity_type,
                    description=e.get("description"),
                    aliases=e.get("aliases", []),
                    context=text[:500],
                ))

            relations = []
            for r in data.get("relations", []):
                try:
                    relation_type = RelationType(r.get("type", "related_to").lower())
                except ValueError:
                    relation_type = RelationType.OTHER

                relations.append(ExtractedRelation(
                    source_entity=r.get("source", ""),
                    target_entity=r.get("target", ""),
                    relation_type=relation_type,
                    relation_label=r.get("label"),
                ))

            logger.info(
                "Extracted entities and relations",
                entity_count=len(entities),
                relation_count=len(relations),
            )

            return entities, relations

        except json.JSONDecodeError as e:
            logger.error("Failed to parse entity extraction JSON", error=str(e))
            return [], []
        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return [], []

    async def process_document_for_graph(
        self,
        document_id: uuid.UUID,
        chunks: Optional[List[Chunk]] = None,
    ) -> Dict[str, int]:
        """
        Process a document to extract entities and build graph.

        Args:
            document_id: Document to process
            chunks: Optional pre-loaded chunks

        Returns:
            Stats dict with counts
        """
        stats = {"entities": 0, "relations": 0, "mentions": 0}

        # Load chunks if not provided
        if not chunks:
            result = await self.db.execute(
                select(Chunk).where(Chunk.document_id == document_id)
            )
            chunks = result.scalars().all()

        if not chunks:
            logger.warning("No chunks found for document", document_id=str(document_id))
            return stats

        all_entities: Dict[str, ExtractedEntity] = {}
        all_relations: List[ExtractedRelation] = []

        # Extract from each chunk
        for chunk in chunks:
            entities, relations = await self.extract_entities_from_text(
                chunk.content,
                document_id=document_id,
                chunk_id=chunk.id,
            )

            # Merge entities (by normalized name)
            for entity in entities:
                norm_name = self._normalize_entity_name(entity.name)
                if norm_name in all_entities:
                    # Merge aliases
                    existing = all_entities[norm_name]
                    existing.aliases = list(set(existing.aliases + entity.aliases))
                else:
                    all_entities[norm_name] = entity

            all_relations.extend(relations)

        # Store entities in database
        entity_map = {}  # norm_name -> Entity model
        for norm_name, extracted in all_entities.items():
            entity = await self._upsert_entity(extracted)
            entity_map[norm_name] = entity

            # Create mention
            await self._create_mention(
                entity_id=entity.id,
                document_id=document_id,
            )
            stats["entities"] += 1
            stats["mentions"] += 1

        # Store relations
        for relation in all_relations:
            source_norm = self._normalize_entity_name(relation.source_entity)
            target_norm = self._normalize_entity_name(relation.target_entity)

            if source_norm in entity_map and target_norm in entity_map:
                await self._upsert_relation(
                    source_entity_id=entity_map[source_norm].id,
                    target_entity_id=entity_map[target_norm].id,
                    relation_type=relation.relation_type,
                    relation_label=relation.relation_label,
                    document_id=document_id,
                )
                stats["relations"] += 1

        await self.db.commit()

        logger.info(
            "Processed document for knowledge graph",
            document_id=str(document_id),
            **stats,
        )

        return stats

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for deduplication."""
        return name.lower().strip()

    async def _upsert_entity(self, extracted: ExtractedEntity) -> Entity:
        """Create or update entity in database."""
        norm_name = self._normalize_entity_name(extracted.name)

        # Check if exists
        result = await self.db.execute(
            select(Entity).where(
                Entity.name_normalized == norm_name,
                Entity.entity_type == extracted.entity_type,
            )
        )
        entity = result.scalar_one_or_none()

        if entity:
            # Update mention count
            entity.mention_count += 1
            if extracted.description and not entity.description:
                entity.description = extracted.description
            if extracted.aliases:
                existing_aliases = entity.aliases or []
                entity.aliases = list(set(existing_aliases + extracted.aliases))
        else:
            # Create new entity
            entity = Entity(
                name=extracted.name,
                name_normalized=norm_name,
                entity_type=extracted.entity_type,
                description=extracted.description,
                aliases=extracted.aliases if extracted.aliases else None,
                mention_count=1,
            )
            self.db.add(entity)
            await self.db.flush()

        return entity

    async def _create_mention(
        self,
        entity_id: uuid.UUID,
        document_id: uuid.UUID,
        chunk_id: Optional[uuid.UUID] = None,
        context_snippet: Optional[str] = None,
    ) -> EntityMention:
        """Create entity mention record."""
        mention = EntityMention(
            entity_id=entity_id,
            document_id=document_id,
            chunk_id=chunk_id,
            context_snippet=context_snippet,
        )
        self.db.add(mention)
        return mention

    async def _upsert_relation(
        self,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        relation_type: RelationType,
        relation_label: Optional[str] = None,
        document_id: Optional[uuid.UUID] = None,
    ) -> EntityRelation:
        """Create or update relationship."""
        # Check if exists
        result = await self.db.execute(
            select(EntityRelation).where(
                EntityRelation.source_entity_id == source_entity_id,
                EntityRelation.target_entity_id == target_entity_id,
                EntityRelation.relation_type == relation_type,
            )
        )
        relation = result.scalar_one_or_none()

        if relation:
            # Increase weight for repeated observations
            relation.weight += 0.1
        else:
            relation = EntityRelation(
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                relation_type=relation_type,
                relation_label=relation_label,
                document_id=document_id,
            )
            self.db.add(relation)

        return relation

    # -------------------------------------------------------------------------
    # Graph Retrieval
    # -------------------------------------------------------------------------

    async def find_entities_by_query(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Entity]:
        """
        Find entities relevant to a query using semantic search.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of relevant entities
        """
        # First try exact/partial name match
        result = await self.db.execute(
            select(Entity)
            .where(
                or_(
                    Entity.name.ilike(f"%{query}%"),
                    Entity.name_normalized.ilike(f"%{query.lower()}%"),
                )
            )
            .order_by(Entity.mention_count.desc())
            .limit(limit)
        )
        entities = list(result.scalars().all())

        # If not enough results, try embedding search
        if len(entities) < limit:
            # TODO: Implement embedding-based entity search
            pass

        return entities

    async def get_entity_neighborhood(
        self,
        entity_id: uuid.UUID,
        max_hops: int = 2,
        max_neighbors: int = 20,
    ) -> Tuple[List[Entity], List[EntityRelation]]:
        """
        Get entities connected to a given entity within N hops.

        Args:
            entity_id: Starting entity
            max_hops: Maximum graph distance
            max_neighbors: Maximum entities to return

        Returns:
            Tuple of (entities, relations)
        """
        visited_entities: Set[uuid.UUID] = {entity_id}
        collected_relations: List[EntityRelation] = []
        frontier: Set[uuid.UUID] = {entity_id}

        for hop in range(max_hops):
            if not frontier:
                break

            # Get outgoing relations
            result = await self.db.execute(
                select(EntityRelation)
                .options(
                    selectinload(EntityRelation.source_entity),
                    selectinload(EntityRelation.target_entity),
                )
                .where(
                    or_(
                        EntityRelation.source_entity_id.in_(frontier),
                        EntityRelation.target_entity_id.in_(frontier),
                    )
                )
                .order_by(EntityRelation.weight.desc())
                .limit(max_neighbors)
            )
            relations = list(result.scalars().all())

            new_frontier: Set[uuid.UUID] = set()
            for rel in relations:
                collected_relations.append(rel)

                # Add new entities to frontier
                if rel.source_entity_id not in visited_entities:
                    new_frontier.add(rel.source_entity_id)
                    visited_entities.add(rel.source_entity_id)
                if rel.target_entity_id not in visited_entities:
                    new_frontier.add(rel.target_entity_id)
                    visited_entities.add(rel.target_entity_id)

                if len(visited_entities) >= max_neighbors:
                    break

            frontier = new_frontier

        # Fetch all entities
        if visited_entities:
            result = await self.db.execute(
                select(Entity).where(Entity.id.in_(visited_entities))
            )
            entities = list(result.scalars().all())
        else:
            entities = []

        return entities, collected_relations

    async def graph_search(
        self,
        query: str,
        max_hops: int = 2,
        top_k: int = 10,
    ) -> GraphRAGContext:
        """
        Perform graph-enhanced search for a query.

        1. Find entities mentioned in query
        2. Expand through graph relationships
        3. Retrieve related document chunks
        4. Build context for RAG

        Args:
            query: Search query
            max_hops: Graph traversal depth
            top_k: Max results

        Returns:
            GraphRAGContext with entities, relations, and chunks
        """
        # Step 1: Find query entities
        query_entities = await self.find_entities_by_query(query, limit=5)

        if not query_entities:
            logger.debug("No entities found for query", query=query)
            return GraphRAGContext(
                entities=[],
                relations=[],
                chunks=[],
                graph_summary="No relevant entities found in knowledge graph.",
                entity_context="",
            )

        # Step 2: Expand through graph
        all_entities: Dict[uuid.UUID, Entity] = {}
        all_relations: List[EntityRelation] = []

        for entity in query_entities:
            all_entities[entity.id] = entity
            neighbors, relations = await self.get_entity_neighborhood(
                entity.id,
                max_hops=max_hops,
                max_neighbors=top_k,
            )
            for n in neighbors:
                all_entities[n.id] = n
            all_relations.extend(relations)

        # Step 3: Get document chunks for these entities
        entity_ids = list(all_entities.keys())

        result = await self.db.execute(
            select(EntityMention)
            .options(selectinload(EntityMention.chunk))
            .where(EntityMention.entity_id.in_(entity_ids))
            .limit(top_k * 2)
        )
        mentions = result.scalars().all()

        chunks = []
        seen_chunk_ids = set()
        for mention in mentions:
            if mention.chunk and mention.chunk.id not in seen_chunk_ids:
                chunks.append(mention.chunk)
                seen_chunk_ids.add(mention.chunk.id)

        # Step 4: Build context
        graph_summary = self._build_graph_summary(
            list(all_entities.values()),
            all_relations,
        )
        entity_context = self._build_entity_context(
            query_entities,
            list(all_entities.values()),
        )

        return GraphRAGContext(
            entities=list(all_entities.values()),
            relations=all_relations,
            chunks=chunks[:top_k],
            graph_summary=graph_summary,
            entity_context=entity_context,
        )

    def _build_graph_summary(
        self,
        entities: List[Entity],
        relations: List[EntityRelation],
    ) -> str:
        """Build a text summary of the graph context."""
        if not entities:
            return "No graph context available."

        lines = ["Knowledge Graph Context:"]

        # Group entities by type
        by_type: Dict[str, List[str]] = defaultdict(list)
        for e in entities:
            by_type[e.entity_type.value].append(e.name)

        for etype, names in by_type.items():
            lines.append(f"- {etype.title()}: {', '.join(names[:5])}")
            if len(names) > 5:
                lines.append(f"  (and {len(names) - 5} more)")

        # Add key relationships
        if relations:
            lines.append("\nKey Relationships:")
            for rel in relations[:10]:
                lines.append(
                    f"- {rel.source_entity.name} --[{rel.relation_type.value}]--> {rel.target_entity.name}"
                )

        return "\n".join(lines)

    def _build_entity_context(
        self,
        primary_entities: List[Entity],
        all_entities: List[Entity],
    ) -> str:
        """Build context string about entities."""
        lines = []

        for entity in primary_entities[:5]:
            line = f"{entity.name} ({entity.entity_type.value})"
            if entity.description:
                line += f": {entity.description}"
            lines.append(line)

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Hybrid Retrieval
    # -------------------------------------------------------------------------

    async def hybrid_search(
        self,
        query: str,
        vector_results: List[Tuple[Chunk, float]],
        graph_weight: float = 0.3,
        top_k: int = 10,
    ) -> List[Tuple[Chunk, float, Optional[GraphRAGContext]]]:
        """
        Combine vector search results with graph-based retrieval.

        Args:
            query: Search query
            vector_results: Results from vector search (chunk, score)
            graph_weight: Weight for graph-based results (0-1)
            top_k: Max results

        Returns:
            List of (chunk, combined_score, graph_context)
        """
        # Get graph context
        graph_context = await self.graph_search(query, max_hops=2, top_k=top_k)

        # Build chunk score map from vector results
        chunk_scores: Dict[uuid.UUID, float] = {}
        chunk_map: Dict[uuid.UUID, Chunk] = {}

        for chunk, score in vector_results:
            chunk_scores[chunk.id] = score * (1 - graph_weight)
            chunk_map[chunk.id] = chunk

        # Boost chunks that appear in graph context
        graph_chunk_ids = {c.id for c in graph_context.chunks}
        for chunk_id in graph_chunk_ids:
            if chunk_id in chunk_scores:
                # Boost existing chunk
                chunk_scores[chunk_id] += graph_weight
            else:
                # Add new chunk from graph
                for gc in graph_context.chunks:
                    if gc.id == chunk_id:
                        chunk_scores[chunk_id] = graph_weight
                        chunk_map[chunk_id] = gc
                        break

        # Sort by combined score
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for chunk_id, score in sorted_chunks:
            chunk = chunk_map.get(chunk_id)
            if chunk:
                results.append((chunk, score, graph_context))

        return results

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        entity_count = await self.db.scalar(select(func.count(Entity.id)))
        relation_count = await self.db.scalar(select(func.count(EntityRelation.id)))
        mention_count = await self.db.scalar(select(func.count(EntityMention.id)))

        # Entity type distribution
        result = await self.db.execute(
            select(Entity.entity_type, func.count(Entity.id))
            .group_by(Entity.entity_type)
        )
        type_dist = {row[0].value: row[1] for row in result}

        return {
            "total_entities": entity_count,
            "total_relations": relation_count,
            "total_mentions": mention_count,
            "entity_type_distribution": type_dist,
        }


# =============================================================================
# Factory Function
# =============================================================================

async def get_knowledge_graph_service(db_session: AsyncSession) -> KnowledgeGraphService:
    """Get configured knowledge graph service."""
    return KnowledgeGraphService(db_session)
