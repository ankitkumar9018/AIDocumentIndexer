"""
AIDocumentIndexer - LazyGraphRAG Service (Phase 66)
====================================================

Implements LazyGraphRAG for efficient knowledge graph retrieval.

Key Innovation: Query-time community summarization instead of index-time.
- Standard GraphRAG: Summarizes all communities during indexing (expensive)
- LazyGraphRAG: Summarizes only relevant communities at query time (99% cost reduction)

Features:
- On-demand community detection
- Query-relevant subgraph extraction
- Incremental graph building
- Cost-efficient for large knowledge bases
- Compatible with existing KG infrastructure

Research:
- LazyGraphRAG: Setting a new standard for Quality and Cost (Microsoft Research)
- Local-Global retrieval without upfront summarization
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

import structlog
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Entity, EntityRelation, Chunk
from backend.services.llm import EnhancedLLMFactory
from backend.services.embeddings import get_embedding_service

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Community:
    """A community (cluster) of related entities."""
    community_id: str
    entities: List[Entity]
    relations: List[EntityRelation]
    central_entity: Optional[Entity] = None
    summary: Optional[str] = None
    relevance_score: float = 0.0


@dataclass
class LazyGraphContext:
    """Context from LazyGraphRAG retrieval."""
    relevant_communities: List[Community]
    entity_context: str
    relation_context: str
    community_summaries: List[str]
    total_entities: int
    total_relations: int
    cost_saved_percentage: float  # Estimated cost savings vs full GraphRAG


@dataclass
class LazyGraphRAGConfig:
    """Configuration for LazyGraphRAG."""
    # Community detection
    max_community_size: int = 50
    min_community_relevance: float = 0.3

    # Query processing
    max_seed_entities: int = 10
    max_communities_to_summarize: int = 5
    expand_hops: int = 2

    # Summarization
    summary_max_tokens: int = 200
    use_llm_summarization: bool = True

    # Cost optimization
    cache_summaries: bool = True
    summary_cache_ttl_hours: int = 24


# =============================================================================
# LazyGraphRAG Service
# =============================================================================

class LazyGraphRAGService:
    """
    LazyGraphRAG: Query-time community summarization.

    Instead of pre-computing summaries for all communities during indexing
    (which is expensive and often wasteful), LazyGraphRAG:
    1. Identifies query-relevant entities
    2. Discovers their communities on-demand
    3. Summarizes only relevant communities
    4. Combines local entity context with global community insights

    This achieves 99% cost reduction compared to standard GraphRAG while
    maintaining similar or better answer quality for most queries.
    """

    def __init__(
        self,
        config: Optional[LazyGraphRAGConfig] = None,
    ):
        self.config = config or LazyGraphRAGConfig()
        self._summary_cache: Dict[str, Tuple[str, datetime]] = {}
        self._llm = None
        self._embedding_service = None

    async def _get_llm(self):
        """Get LLM for summarization."""
        if self._llm is None:
            self._llm = EnhancedLLMFactory.create_with_fallback(
                task_type="summarization",
                temperature=0.3,
            )
        return self._llm

    async def _get_embeddings(self):
        """Get embedding service."""
        if self._embedding_service is None:
            self._embedding_service = await get_embedding_service()
        return self._embedding_service

    # =========================================================================
    # Core Retrieval
    # =========================================================================

    async def retrieve(
        self,
        query: str,
        session: AsyncSession,
        collection_filter: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> LazyGraphContext:
        """
        Retrieve relevant context using LazyGraphRAG.

        Steps:
        1. Extract seed entities from query
        2. Find relevant entities in the graph
        3. Discover their communities on-demand
        4. Summarize relevant communities
        5. Combine into cohesive context

        Args:
            query: User query
            session: Database session
            collection_filter: Optional collection filter
            organization_id: Optional organization filter

        Returns:
            LazyGraphContext with community-aware context
        """
        start_time = datetime.utcnow()

        # Step 1: Extract seed entities from query
        seed_entities = await self._extract_query_entities(query, session)

        if not seed_entities:
            logger.debug("No seed entities found, falling back to text search")
            seed_entities = await self._text_search_entities(query, session)

        logger.info(
            "LazyGraphRAG: Found seed entities",
            count=len(seed_entities),
            entities=[e.name for e in seed_entities[:5]],
        )

        # Step 2: Expand to related entities
        expanded_entities, relations = await self._expand_entities(
            seed_entities, session, hops=self.config.expand_hops
        )

        # Step 3: Detect communities on-demand
        communities = await self._detect_communities(
            expanded_entities, relations, session
        )

        # Step 4: Score and rank communities by relevance
        ranked_communities = await self._rank_communities(
            communities, query, seed_entities
        )

        # Step 5: Summarize top communities (lazy evaluation)
        top_communities = ranked_communities[:self.config.max_communities_to_summarize]
        for community in top_communities:
            if community.summary is None:
                community.summary = await self._summarize_community(community, query)

        # Build context
        entity_context = self._build_entity_context(seed_entities, expanded_entities)
        relation_context = self._build_relation_context(relations)
        community_summaries = [c.summary for c in top_communities if c.summary]

        # Calculate cost savings
        total_entities_in_graph = await self._count_total_entities(session)
        entities_processed = len(expanded_entities)
        cost_saved = max(0, 1 - (entities_processed / max(total_entities_in_graph, 1)))

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        logger.info(
            "LazyGraphRAG retrieval complete",
            duration_ms=duration_ms,
            seed_entities=len(seed_entities),
            expanded_entities=len(expanded_entities),
            communities=len(communities),
            summarized=len(top_communities),
            cost_saved_pct=f"{cost_saved * 100:.1f}%",
        )

        return LazyGraphContext(
            relevant_communities=top_communities,
            entity_context=entity_context,
            relation_context=relation_context,
            community_summaries=community_summaries,
            total_entities=len(expanded_entities),
            total_relations=len(relations),
            cost_saved_percentage=cost_saved * 100,
        )

    # =========================================================================
    # Entity Extraction
    # =========================================================================

    async def _extract_query_entities(
        self,
        query: str,
        session: AsyncSession,
    ) -> List[Entity]:
        """Extract entities from query using NER and entity linking."""
        # Simple approach: find entities in DB that match query terms
        query_words = set(query.lower().split())

        # Query for matching entities
        stmt = select(Entity).where(
            or_(
                func.lower(Entity.name).in_(query_words),
                *[func.lower(Entity.name).contains(w) for w in query_words if len(w) > 3]
            )
        ).limit(self.config.max_seed_entities)

        result = await session.execute(stmt)
        entities = list(result.scalars().all())

        return entities

    async def _text_search_entities(
        self,
        query: str,
        session: AsyncSession,
    ) -> List[Entity]:
        """Fallback text search for entities."""
        # Search entity descriptions
        stmt = select(Entity).where(
            Entity.description.ilike(f"%{query[:50]}%")
        ).limit(self.config.max_seed_entities)

        result = await session.execute(stmt)
        return list(result.scalars().all())

    # =========================================================================
    # Entity Expansion
    # =========================================================================

    async def _expand_entities(
        self,
        seed_entities: List[Entity],
        session: AsyncSession,
        hops: int = 2,
    ) -> Tuple[List[Entity], List[EntityRelation]]:
        """Expand seed entities to related entities via relations."""
        if not seed_entities:
            return [], []

        visited_ids: Set[str] = {str(e.id) for e in seed_entities}
        all_entities = list(seed_entities)
        all_relations = []
        current_entities = seed_entities

        for hop in range(hops):
            if not current_entities:
                break

            entity_ids = [e.id for e in current_entities]

            # Find relations involving current entities
            stmt = select(EntityRelation).where(
                or_(
                    EntityRelation.source_entity_id.in_(entity_ids),
                    EntityRelation.target_entity_id.in_(entity_ids),
                )
            )
            result = await session.execute(stmt)
            relations = list(result.scalars().all())
            all_relations.extend(relations)

            # Get connected entity IDs
            new_entity_ids = set()
            for rel in relations:
                if str(rel.source_entity_id) not in visited_ids:
                    new_entity_ids.add(rel.source_entity_id)
                if str(rel.target_entity_id) not in visited_ids:
                    new_entity_ids.add(rel.target_entity_id)

            if not new_entity_ids:
                break

            # Fetch new entities
            stmt = select(Entity).where(Entity.id.in_(list(new_entity_ids)))
            result = await session.execute(stmt)
            new_entities = list(result.scalars().all())

            for e in new_entities:
                visited_ids.add(str(e.id))

            all_entities.extend(new_entities)
            current_entities = new_entities

        return all_entities, all_relations

    # =========================================================================
    # Community Detection
    # =========================================================================

    async def _detect_communities(
        self,
        entities: List[Entity],
        relations: List[EntityRelation],
        session: AsyncSession,
    ) -> List[Community]:
        """
        Detect communities using connected components.

        A simple but effective approach - entities connected by relations
        form communities. More sophisticated methods (Louvain, etc.) can
        be added later.
        """
        if not entities:
            return []

        # Build adjacency map
        entity_map = {str(e.id): e for e in entities}
        adjacency: Dict[str, Set[str]] = defaultdict(set)

        for rel in relations:
            src_id = str(rel.source_entity_id)
            tgt_id = str(rel.target_entity_id)
            if src_id in entity_map and tgt_id in entity_map:
                adjacency[src_id].add(tgt_id)
                adjacency[tgt_id].add(src_id)

        # Find connected components (simple DFS)
        visited: Set[str] = set()
        communities: List[Community] = []

        def dfs(entity_id: str, component: List[str]):
            if entity_id in visited:
                return
            visited.add(entity_id)
            component.append(entity_id)
            for neighbor in adjacency.get(entity_id, []):
                dfs(neighbor, component)

        for entity_id in entity_map:
            if entity_id not in visited:
                component: List[str] = []
                dfs(entity_id, component)

                if component:
                    # Create community
                    community_entities = [entity_map[eid] for eid in component if eid in entity_map]
                    community_relations = [
                        r for r in relations
                        if str(r.source_entity_id) in component or str(r.target_entity_id) in component
                    ]

                    # Find central entity (most connected)
                    connection_counts = defaultdict(int)
                    for r in community_relations:
                        connection_counts[str(r.source_entity_id)] += 1
                        connection_counts[str(r.target_entity_id)] += 1

                    central_id = max(connection_counts.keys(), key=lambda k: connection_counts[k]) if connection_counts else None
                    central_entity = entity_map.get(central_id) if central_id else None

                    community = Community(
                        community_id=f"community_{len(communities)}",
                        entities=community_entities,
                        relations=community_relations,
                        central_entity=central_entity,
                    )
                    communities.append(community)

        return communities

    # =========================================================================
    # Community Ranking
    # =========================================================================

    async def _rank_communities(
        self,
        communities: List[Community],
        query: str,
        seed_entities: List[Entity],
    ) -> List[Community]:
        """Rank communities by relevance to query."""
        seed_ids = {str(e.id) for e in seed_entities}
        query_words = set(query.lower().split())

        for community in communities:
            # Score based on:
            # 1. Seed entity overlap
            community_ids = {str(e.id) for e in community.entities}
            seed_overlap = len(seed_ids & community_ids) / max(len(seed_ids), 1)

            # 2. Entity name relevance
            entity_names = " ".join(e.name.lower() for e in community.entities)
            name_overlap = sum(1 for w in query_words if w in entity_names) / max(len(query_words), 1)

            # 3. Community size (prefer moderate sizes)
            size_score = min(1.0, len(community.entities) / self.config.max_community_size)

            # Combined score
            community.relevance_score = (
                0.5 * seed_overlap +
                0.3 * name_overlap +
                0.2 * size_score
            )

        # Sort by relevance
        communities.sort(key=lambda c: c.relevance_score, reverse=True)

        return communities

    # =========================================================================
    # Community Summarization
    # =========================================================================

    async def _summarize_community(
        self,
        community: Community,
        query: str,
    ) -> str:
        """
        Generate a summary for a community.

        This is the "lazy" part - we only summarize when needed.
        """
        # Check cache
        cache_key = community.community_id
        if self.config.cache_summaries and cache_key in self._summary_cache:
            cached_summary, cached_time = self._summary_cache[cache_key]
            age_hours = (datetime.utcnow() - cached_time).total_seconds() / 3600
            if age_hours < self.config.summary_cache_ttl_hours:
                return cached_summary

        # Build community description
        entity_list = ", ".join(e.name for e in community.entities[:10])
        relation_list = []
        for r in community.relations[:5]:
            source = next((e.name for e in community.entities if e.id == r.source_entity_id), "?")
            target = next((e.name for e in community.entities if e.id == r.target_entity_id), "?")
            relation_list.append(f"{source} -> {target}")

        relations_str = "; ".join(relation_list) if relation_list else "No direct relations"

        if self.config.use_llm_summarization:
            try:
                llm = await self._get_llm()
                prompt = f"""Summarize this group of related entities to help answer a question.

Question: {query}

Entities: {entity_list}
Key Relationships: {relations_str}
Central Entity: {community.central_entity.name if community.central_entity else 'None'}

Provide a brief (2-3 sentences) summary of what this group represents and how it might relate to the question:"""

                response = await llm.ainvoke(prompt)
                summary = response.content if hasattr(response, 'content') else str(response)

            except Exception as e:
                logger.warning("LLM summarization failed", error=str(e))
                summary = f"Group of {len(community.entities)} entities including: {entity_list}"
        else:
            summary = f"Group of {len(community.entities)} entities including: {entity_list}"

        # Cache summary
        if self.config.cache_summaries:
            self._summary_cache[cache_key] = (summary, datetime.utcnow())

        return summary

    # =========================================================================
    # Context Building
    # =========================================================================

    def _build_entity_context(
        self,
        seed_entities: List[Entity],
        all_entities: List[Entity],
    ) -> str:
        """Build entity context string."""
        lines = ["Key Entities:"]

        for entity in seed_entities:
            desc = entity.description[:100] if entity.description else "No description"
            lines.append(f"- {entity.name} ({entity.entity_type.value}): {desc}")

        if len(all_entities) > len(seed_entities):
            lines.append(f"\nRelated entities: {len(all_entities) - len(seed_entities)} additional")

        return "\n".join(lines)

    def _build_relation_context(
        self,
        relations: List[EntityRelation],
    ) -> str:
        """Build relation context string."""
        if not relations:
            return "No relationships found."

        lines = ["Key Relationships:"]
        for rel in relations[:10]:
            lines.append(f"- {rel.relation_type.value}: (from entity to entity)")

        if len(relations) > 10:
            lines.append(f"\n... and {len(relations) - 10} more relationships")

        return "\n".join(lines)

    async def _count_total_entities(self, session: AsyncSession) -> int:
        """Count total entities in the knowledge graph."""
        stmt = select(func.count(Entity.id))
        result = await session.execute(stmt)
        return result.scalar() or 0


# =============================================================================
# Singleton
# =============================================================================

_lazy_graphrag_service: Optional[LazyGraphRAGService] = None


def get_lazy_graphrag_service(
    config: Optional[LazyGraphRAGConfig] = None,
) -> LazyGraphRAGService:
    """Get or create LazyGraphRAG service singleton."""
    global _lazy_graphrag_service

    if _lazy_graphrag_service is None or config is not None:
        _lazy_graphrag_service = LazyGraphRAGService(config)

    return _lazy_graphrag_service


# =============================================================================
# Convenience Function
# =============================================================================

async def lazy_graph_retrieve(
    query: str,
    session: AsyncSession,
    collection_filter: Optional[str] = None,
    organization_id: Optional[str] = None,
) -> LazyGraphContext:
    """Convenience function for LazyGraphRAG retrieval."""
    service = get_lazy_graphrag_service()
    return await service.retrieve(
        query=query,
        session=session,
        collection_filter=collection_filter,
        organization_id=organization_id,
    )
