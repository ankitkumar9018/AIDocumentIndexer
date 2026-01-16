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
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import structlog
from sqlalchemy import select, func, and_, or_, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from unidecode import unidecode  # Unicode to ASCII transliteration

from backend.db.models import (
    Entity, EntityMention, EntityRelation,
    EntityType, RelationType,
    Document, Chunk, AccessTier,
)
from backend.services.llm import EnhancedLLMFactory
from backend.services.embeddings import get_embedding_service

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractedEntity:
    """Entity extracted from text with language awareness."""
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    context: Optional[str] = None
    # Language-aware fields for cross-language entity linking
    language: Optional[str] = None  # Source document language: "en", "de", "ru", etc.
    canonical_name: Optional[str] = None  # English canonical form for linking
    language_variants: Dict[str, str] = field(default_factory=dict)  # {"en": "Germany", "de": "Deutschland"}


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

# Language name mapping for prompts
LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
}

ENTITY_EXTRACTION_PROMPT = """Extract named entities and their relationships from the following text.

SOURCE DOCUMENT LANGUAGE: {document_language}

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
1. name: The entity name AS IT APPEARS in the source text
2. type: One of the types above
3. description: Brief description if available from context
4. aliases: Alternative names, abbreviations, OR translations in other languages
5. canonical_name: The ENGLISH name for this entity (for cross-language linking)
   - For "Deutschland" → canonical_name: "Germany"
   - For "Москва" → canonical_name: "Moscow"
   - For "東京" → canonical_name: "Tokyo"
   - For person names, keep original spelling: "François Hollande" → "François Hollande"
   - For product/company names, keep original: "BMW" → "BMW"

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
    {{
      "name": "Deutschland",
      "type": "LOCATION",
      "description": "Country in central Europe",
      "aliases": ["BRD", "Federal Republic of Germany", "Allemagne"],
      "canonical_name": "Germany"
    }}
  ],
  "relations": [
    {{"source": "entity_name", "target": "entity_name", "type": "WORKS_FOR|...", "label": "optional label"}}
  ]
}}

Only include entities and relations clearly supported by the text. Be precise and avoid speculation."""


def _extract_json_from_response(response: str) -> Optional[dict]:
    """
    Extract JSON from LLM response with multiple fallback strategies.

    Handles common LLM response formats:
    - JSON wrapped in markdown code blocks
    - JSON with explanatory text before/after
    - Multiple JSON objects (takes first valid one)

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON dict or None if extraction fails
    """
    if not response:
        return None

    # Strategy 1: Remove markdown code blocks
    clean = re.sub(r'```(?:json)?\s*', '', response)
    clean = re.sub(r'```', '', clean)
    clean = clean.strip()

    # Strategy 2: Find matching braces (handles nested JSON properly)
    start = clean.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(clean[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                json_str = clean[start:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try with some cleanup
                    try:
                        # Remove trailing commas before ] or }
                        cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        return None

    return None


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
        Get or initialize LLM service with graceful fallback.

        Uses the configured provider from environment or database.
        Supports FREE (Ollama) and PAID (OpenAI, Anthropic) providers.

        Returns:
            LLM instance or None if unavailable
        """
        if not self.llm:
            try:
                llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                    operation="chat",  # Use chat operation for entity extraction
                    db_session=self.db,
                )
                self.llm = llm
            except Exception as e:
                logger.warning(
                    "Failed to initialize LLM for knowledge graph",
                    error=str(e),
                )
                return None
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
        document_language: str = "en",
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        Extract entities and relationships from text using LLM.

        Args:
            text: Text to extract from
            document_id: Source document ID
            chunk_id: Source chunk ID
            document_language: Language code of the source document (e.g., "en", "de", "ru")

        Returns:
            Tuple of (entities, relations)
        """
        llm = await self._get_llm()

        # Graceful fallback: Return empty if LLM unavailable
        if not llm:
            logger.warning(
                "LLM not available for entity extraction - skipping",
                document_id=str(document_id) if document_id else None,
            )
            return [], []

        # Get human-readable language name for the prompt
        language_name = LANGUAGE_NAMES.get(document_language, "English")

        prompt = ENTITY_EXTRACTION_PROMPT.format(
            text=text[:8000],
            document_language=language_name,
        )

        try:
            # Use langchain model interface
            result = await llm.ainvoke(prompt)
            response = result.content if hasattr(result, 'content') else str(result)

            # Parse JSON response using robust extraction
            data = _extract_json_from_response(response)
            if not data:
                logger.warning(
                    "No valid JSON found in entity extraction response",
                    response_preview=response[:200] if response else "empty"
                )
                return [], []

            entities = []
            for e in data.get("entities", []):
                try:
                    entity_type = EntityType(e.get("type", "other").lower())
                except ValueError:
                    entity_type = EntityType.OTHER

                # Get canonical name from LLM response, fallback to original name
                canonical_name = e.get("canonical_name") or e.get("name", "")

                entities.append(ExtractedEntity(
                    name=e.get("name", ""),
                    entity_type=entity_type,
                    description=e.get("description"),
                    aliases=e.get("aliases", []),
                    context=text[:500],
                    language=document_language,
                    canonical_name=canonical_name,
                    language_variants={document_language: e.get("name", "")} if document_language else {},
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

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return [], []

    async def extract_entities_batch(
        self,
        chunks: List[str],
        document_id: Optional[uuid.UUID] = None,
        document_language: str = "en",
        batch_size: int = 3,
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        Extract entities from multiple text chunks in batches for better performance.

        Instead of making one LLM call per chunk, this method combines multiple
        chunks into single prompts, reducing LLM calls by ~3-5x.

        Args:
            chunks: List of text chunks to process
            document_id: Source document ID
            document_language: Language code of the source document
            batch_size: Number of chunks to combine per LLM call

        Returns:
            Tuple of (all_entities, all_relations) from all chunks
        """
        llm = await self._get_llm()

        # Graceful fallback: Return empty if LLM unavailable
        if not llm:
            logger.warning(
                "LLM not available for batch entity extraction - skipping",
                document_id=str(document_id) if document_id else None,
                chunk_count=len(chunks),
            )
            return [], []

        all_entities: List[ExtractedEntity] = []
        all_relations: List[ExtractedRelation] = []
        language_name = LANGUAGE_NAMES.get(document_language, "English")

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Combine chunks with separators
            combined_text = "\n\n---CHUNK_BOUNDARY---\n\n".join(
                f"[Chunk {j+1}]\n{chunk[:2500]}" for j, chunk in enumerate(batch)
            )

            prompt = f"""Extract named entities and relationships from the following text chunks.

SOURCE DOCUMENT LANGUAGE: {language_name}

The text is divided into {len(batch)} chunks separated by ---CHUNK_BOUNDARY---.
Extract entities from ALL chunks.

Text:
{combined_text}

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
1. name: The entity name AS IT APPEARS in the source text
2. type: One of the types above
3. description: Brief description if available from context
4. canonical_name: The ENGLISH name for this entity (for cross-language linking)

Return JSON in this exact format:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "...", "canonical_name": "..."}}
  ],
  "relations": [
    {{"source": "entity_name", "target": "entity_name", "type": "WORKS_FOR|LOCATED_IN|RELATED_TO|PART_OF|...", "label": "optional"}}
  ]
}}

Only include entities and relations clearly supported by the text."""

            try:
                result = await llm.ainvoke(prompt)
                response = result.content if hasattr(result, 'content') else str(result)

                data = _extract_json_from_response(response)
                if not data:
                    logger.warning(
                        "No valid JSON in batch extraction response",
                        batch_index=i // batch_size,
                    )
                    continue

                # Parse entities
                for e in data.get("entities", []):
                    try:
                        entity_type = EntityType(e.get("type", "other").lower())
                    except ValueError:
                        entity_type = EntityType.OTHER

                    canonical_name = e.get("canonical_name") or e.get("name", "")

                    all_entities.append(ExtractedEntity(
                        name=e.get("name", ""),
                        entity_type=entity_type,
                        description=e.get("description"),
                        aliases=e.get("aliases", []),
                        language=document_language,
                        canonical_name=canonical_name,
                        language_variants={document_language: e.get("name", "")} if document_language else {},
                    ))

                # Parse relations
                for r in data.get("relations", []):
                    try:
                        relation_type = RelationType(r.get("type", "related_to").lower())
                    except ValueError:
                        relation_type = RelationType.OTHER

                    all_relations.append(ExtractedRelation(
                        source_entity=r.get("source", ""),
                        target_entity=r.get("target", ""),
                        relation_type=relation_type,
                        relation_label=r.get("label"),
                    ))

                logger.debug(
                    "Batch extraction completed",
                    batch_index=i // batch_size,
                    chunks_in_batch=len(batch),
                    entities_found=len(data.get("entities", [])),
                )

            except Exception as e:
                logger.error(
                    "Batch entity extraction failed",
                    batch_index=i // batch_size,
                    error=str(e),
                )
                continue

        logger.info(
            "Batch entity extraction completed",
            total_chunks=len(chunks),
            total_entities=len(all_entities),
            total_relations=len(all_relations),
            batches_processed=(len(chunks) + batch_size - 1) // batch_size,
        )

        return all_entities, all_relations

    async def process_document_for_graph(
        self,
        document_id: uuid.UUID,
        chunks: Optional[List[Chunk]] = None,
        use_batch_extraction: bool = True,
    ) -> Dict[str, int]:
        """
        Process a document to extract entities and build graph with language context.

        Loads the document language and passes it to entity extraction for
        cross-language entity linking support. Uses batch extraction by default
        for 3-5x faster processing.

        Args:
            document_id: Document to process
            chunks: Optional pre-loaded chunks
            use_batch_extraction: Use batch extraction for better performance (default True)

        Returns:
            Stats dict with counts
        """
        stats = {"entities": 0, "relations": 0, "mentions": 0}

        # Load document to get language
        doc_result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = doc_result.scalar_one_or_none()

        if not document:
            logger.warning("Document not found", document_id=str(document_id))
            return stats

        # Get document language (default to English if not set)
        document_language = document.language or "en"

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

        # Use batch extraction for better performance (3-5x faster)
        if use_batch_extraction and len(chunks) > 1:
            chunk_texts = [chunk.content for chunk in chunks]
            entities, relations = await self.extract_entities_batch(
                chunks=chunk_texts,
                document_id=document_id,
                document_language=document_language,
                batch_size=3,  # Process 3 chunks per LLM call
            )

            # Merge entities (by normalized name)
            for entity in entities:
                norm_name = self._normalize_entity_name(entity.name)
                if norm_name in all_entities:
                    existing = all_entities[norm_name]
                    existing.aliases = list(set(existing.aliases + entity.aliases))
                else:
                    all_entities[norm_name] = entity

            all_relations.extend(relations)
        else:
            # Fallback to sequential extraction for single chunk or when batch disabled
            for chunk in chunks:
                entities, relations = await self.extract_entities_from_text(
                    chunk.content,
                    document_id=document_id,
                    chunk_id=chunk.id,
                    document_language=document_language,
                )

                # Merge entities (by normalized name)
                for entity in entities:
                    norm_name = self._normalize_entity_name(entity.name)
                    if norm_name in all_entities:
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

            # Create mention with language context
            await self._create_mention(
                entity_id=entity.id,
                document_id=document_id,
                mention_language=document_language,
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

    def _normalize_entity_name(self, name: str, preserve_script: bool = False) -> str:
        """
        Normalize entity name with Unicode awareness for deduplication.

        Steps:
        1. Unicode NFC normalization
        2. Case folding (better than lower() for Unicode)
        3. Whitespace normalization
        4. Optional: ASCII transliteration for cross-script matching

        Args:
            name: Entity name to normalize
            preserve_script: If True, keep original script; if False, transliterate to ASCII

        Returns:
            Normalized entity name
        """
        if not name:
            return ""

        # Step 1: Unicode NFC normalization (composed form)
        normalized = unicodedata.normalize('NFC', name)

        # Step 2: Case fold (handles ß → ss, İ → i, etc.)
        normalized = normalized.casefold()

        # Step 3: Collapse whitespace
        normalized = ' '.join(normalized.split())

        # Step 4: Optionally transliterate to ASCII for matching
        if not preserve_script:
            # "Müller" → "muller", "東京" → "dong jing", "Москва" → "moskva"
            ascii_version = unidecode(normalized)
            return ascii_version.strip()

        return normalized.strip()

    def _get_ascii_normalized(self, name: str) -> str:
        """Get ASCII-safe version for cross-script matching."""
        if not name:
            return ""
        return unidecode(name.casefold()).strip()

    async def _upsert_entity(self, extracted: ExtractedEntity) -> Entity:
        """
        Create or update entity in database with language-aware matching.

        Uses multi-strategy matching:
        1. Exact normalized name match (same language)
        2. Canonical name match (cross-language linking)
        3. ASCII-normalized match (cross-script matching)
        """
        norm_name = self._normalize_entity_name(extracted.name)
        canonical = extracted.canonical_name or extracted.name
        canonical_norm = self._normalize_entity_name(canonical)

        entity = None

        # Strategy 1: Try exact normalized match (most precise)
        result = await self.db.execute(
            select(Entity).where(
                Entity.name_normalized == norm_name,
                Entity.entity_type == extracted.entity_type,
            )
        )
        entity = result.scalar_one_or_none()

        # Strategy 2: Try canonical name match (cross-language linking)
        if not entity and canonical_norm != norm_name:
            result = await self.db.execute(
                select(Entity).where(
                    or_(
                        Entity.canonical_name == canonical,
                        Entity.name_normalized == canonical_norm,
                    ),
                    Entity.entity_type == extracted.entity_type,
                )
            )
            entity = result.scalar_one_or_none()

        # Strategy 3: Try ASCII-normalized match (cross-script)
        if not entity:
            ascii_canonical = self._get_ascii_normalized(canonical)
            result = await self.db.execute(
                select(Entity).where(
                    func.lower(Entity.canonical_name) == ascii_canonical,
                    Entity.entity_type == extracted.entity_type,
                )
            )
            entity = result.scalar_one_or_none()

        if entity:
            # Update existing entity
            entity.mention_count += 1

            # Update description if not set
            if extracted.description and not entity.description:
                entity.description = extracted.description

            # Merge language variant
            variants = entity.language_variants or {}
            if extracted.language and extracted.name:
                variants[extracted.language] = extracted.name
            entity.language_variants = variants

            # Merge aliases
            if extracted.aliases:
                existing_aliases = entity.aliases or []
                entity.aliases = list(set(existing_aliases + extracted.aliases))

            # Update canonical name if not set
            if extracted.canonical_name and not entity.canonical_name:
                entity.canonical_name = extracted.canonical_name
        else:
            # Create new entity with language support
            entity = Entity(
                name=extracted.name,
                name_normalized=norm_name,
                entity_type=extracted.entity_type,
                description=extracted.description,
                aliases=extracted.aliases if extracted.aliases else None,
                entity_language=extracted.language,
                canonical_name=canonical,
                language_variants={extracted.language: extracted.name} if extracted.language else None,
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
        mention_language: Optional[str] = None,
        mention_script: Optional[str] = None,
    ) -> EntityMention:
        """Create entity mention record with language context."""
        mention = EntityMention(
            entity_id=entity_id,
            document_id=document_id,
            chunk_id=chunk_id,
            context_snippet=context_snippet,
            mention_language=mention_language,
            mention_script=mention_script,
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
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10,
        query_language: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> List[Entity]:
        """
        Find entities relevant to a query with language-aware matching.

        Uses multi-strategy matching:
        1. Exact normalized name match
        2. Canonical name match (cross-language)
        3. Name contains query (substring)
        4. Alias match
        5. Language variant match

        Args:
            query: Search query
            entity_types: Optional filter by entity types
            limit: Max results
            query_language: Optional language code for variant search
            organization_id: Optional organization ID for multi-tenant isolation

        Returns:
            List of relevant entities, prioritized by match quality
        """
        query_norm = self._normalize_entity_name(query)
        query_ascii = self._get_ascii_normalized(query)

        # Build multi-strategy matching conditions
        conditions = [
            # Strategy 1: Exact normalized match
            Entity.name_normalized == query_norm,
            # Strategy 2: Canonical name match (cross-language)
            func.lower(Entity.canonical_name) == query_norm,
            # Strategy 3: ASCII-normalized canonical match
            func.lower(Entity.canonical_name) == query_ascii,
            # Strategy 4: Name contains query (substring)
            Entity.name_normalized.ilike(f"%{query_norm}%"),
            # Strategy 5: Query in aliases (if PostgreSQL array)
            Entity.aliases.any(query),
        ]

        # Add language variant search if query_language specified
        if query_language:
            # Search in language_variants JSON field
            conditions.append(
                Entity.language_variants[query_language].astext.ilike(f"%{query}%")
            )

        base_query = select(Entity).where(or_(*conditions))

        # Filter by organization for multi-tenant isolation
        if organization_id:
            org_uuid = uuid.UUID(organization_id)
            base_query = base_query.where(Entity.organization_id == org_uuid)

        # Optional entity type filter
        if entity_types:
            base_query = base_query.where(Entity.entity_type.in_(entity_types))

        # Order by match quality and mention count
        result = await self.db.execute(
            base_query.order_by(
                # Prioritize exact matches over partial matches
                case(
                    (Entity.name_normalized == query_norm, 1),
                    (func.lower(Entity.canonical_name) == query_norm, 2),
                    (func.lower(Entity.canonical_name) == query_ascii, 3),
                    else_=4,
                ),
                Entity.mention_count.desc(),
            ).limit(limit)
        )
        entities = list(result.scalars().all())

        # If not enough results, try embedding search
        if len(entities) < limit:
            # TODO: Implement embedding-based entity search
            pass

        logger.debug(
            "Entity search completed",
            query=query,
            query_language=query_language,
            results_count=len(entities),
        )

        return entities

    async def search_entities_semantic(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        organization_id: Optional[str] = None,
    ) -> List[Tuple[Entity, float]]:
        """
        Search entities using embedding similarity.

        This enables semantic search over entities when text-based matching fails,
        allowing the system to find conceptually related entities even when
        names don't match exactly.

        Args:
            query: Search query
            entity_types: Optional filter by entity types
            top_k: Maximum results to return
            similarity_threshold: Minimum similarity score (0-1)
            organization_id: Optional organization ID for multi-tenant isolation

        Returns:
            List of (entity, similarity_score) tuples sorted by similarity
        """
        from backend.services.embeddings import EmbeddingService, compute_similarity

        # Generate query embedding
        embedding_service = get_embedding_service(use_ray=False)
        query_embedding = embedding_service.embed_text(query)

        if not query_embedding or all(v == 0 for v in query_embedding):
            logger.warning("Failed to generate query embedding for semantic search")
            return []

        # Build base query for entities with embeddings
        base_query = select(Entity).where(Entity.embedding.isnot(None))

        # Filter by organization for multi-tenant isolation
        if organization_id:
            org_uuid = uuid.UUID(organization_id)
            base_query = base_query.where(Entity.organization_id == org_uuid)

        if entity_types:
            base_query = base_query.where(Entity.entity_type.in_(entity_types))

        result = await self.db.execute(base_query)
        entities = list(result.scalars().all())

        if not entities:
            logger.debug("No entities with embeddings found for semantic search")
            return []

        # Calculate similarity scores
        scored_entities = []
        for entity in entities:
            if entity.embedding:
                # Handle both list and string (JSON) storage formats
                entity_embedding = entity.embedding
                if isinstance(entity_embedding, str):
                    import json
                    entity_embedding = json.loads(entity_embedding)

                similarity = compute_similarity(query_embedding, entity_embedding)
                if similarity >= similarity_threshold:
                    scored_entities.append((entity, similarity))

        # Sort by similarity (descending) and limit
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        results = scored_entities[:top_k]

        logger.debug(
            "Semantic entity search completed",
            query=query[:50],
            candidates=len(entities),
            matches=len(results),
            threshold=similarity_threshold,
        )

        return results

    async def find_entities_hybrid(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10,
        query_language: Optional[str] = None,
        semantic_weight: float = 0.5,
        organization_id: Optional[str] = None,
    ) -> List[Tuple[Entity, float]]:
        """
        Hybrid entity search combining text-based and semantic matching.

        Combines:
        1. Text-based matching (exact, normalized, alias)
        2. Embedding-based semantic similarity

        Args:
            query: Search query
            entity_types: Optional filter by entity types
            limit: Maximum results
            query_language: Optional language code
            semantic_weight: Weight for semantic scores (0-1), text weight = 1 - semantic_weight
            organization_id: Optional organization ID for multi-tenant isolation

        Returns:
            List of (entity, combined_score) tuples
        """
        # Get text-based matches
        text_entities = await self.find_entities_by_query(
            query=query,
            entity_types=entity_types,
            limit=limit * 2,  # Get more candidates for merging
            query_language=query_language,
            organization_id=organization_id,
        )

        # Get semantic matches
        semantic_results = await self.search_entities_semantic(
            query=query,
            entity_types=entity_types,
            top_k=limit * 2,
            similarity_threshold=0.5,  # Lower threshold for hybrid
            organization_id=organization_id,
        )

        # Combine results with scoring
        entity_scores: Dict[uuid.UUID, Tuple[Entity, float, float]] = {}

        # Score text matches (position-based scoring)
        for idx, entity in enumerate(text_entities):
            text_score = 1.0 - (idx / len(text_entities)) if text_entities else 0
            entity_scores[entity.id] = (entity, text_score, 0.0)

        # Add/update with semantic scores
        for entity, sem_score in semantic_results:
            if entity.id in entity_scores:
                # Entity found in both - combine scores
                existing = entity_scores[entity.id]
                entity_scores[entity.id] = (entity, existing[1], sem_score)
            else:
                # Only in semantic results
                entity_scores[entity.id] = (entity, 0.0, sem_score)

        # Calculate combined scores
        combined_results = []
        for entity_id, (entity, text_score, sem_score) in entity_scores.items():
            combined_score = (
                (1 - semantic_weight) * text_score +
                semantic_weight * sem_score
            )
            combined_results.append((entity, combined_score))

        # Sort by combined score and limit
        combined_results.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            "Hybrid entity search completed",
            query=query[:50],
            text_matches=len(text_entities),
            semantic_matches=len(semantic_results),
            combined_results=len(combined_results[:limit]),
        )

        return combined_results[:limit]

    async def ensure_entity_embeddings(
        self,
        batch_size: int = 100,
    ) -> int:
        """
        Generate embeddings for entities that don't have them.

        This should be called periodically or after bulk entity extraction
        to ensure all entities have embeddings for semantic search.

        Args:
            batch_size: Number of entities to process per batch

        Returns:
            Number of entities updated with embeddings
        """
        from backend.services.embeddings import EmbeddingService

        # Find entities without embeddings
        result = await self.db.execute(
            select(Entity).where(Entity.embedding.is_(None)).limit(batch_size)
        )
        entities = list(result.scalars().all())

        if not entities:
            logger.debug("All entities have embeddings")
            return 0

        embedding_service = EmbeddingService()

        # Generate text for embedding (name + description + aliases)
        texts = []
        for entity in entities:
            text_parts = [entity.name]
            if entity.description:
                text_parts.append(entity.description)
            if entity.aliases:
                text_parts.extend(entity.aliases[:3])  # Limit aliases
            texts.append(" ".join(text_parts))

        # Batch embed
        embeddings = embedding_service.embed_texts(texts)

        # Update entities
        updated = 0
        for entity, embedding in zip(entities, embeddings):
            if embedding and not all(v == 0 for v in embedding):
                entity.embedding = embedding
                updated += 1

        await self.db.commit()

        logger.info(
            "Generated entity embeddings",
            total_entities=len(entities),
            updated=updated,
        )

        return updated

    async def get_entity_neighborhood(
        self,
        entity_id: uuid.UUID,
        max_hops: int = 2,
        max_neighbors: int = 20,
        organization_id: Optional[str] = None,
    ) -> Tuple[List[Entity], List[EntityRelation]]:
        """
        Get entities connected to a given entity within N hops.

        Args:
            entity_id: Starting entity
            max_hops: Maximum graph distance
            max_neighbors: Maximum entities to return
            organization_id: Optional organization ID for multi-tenant isolation

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
            relation_query = (
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
            )

            # Filter by organization for multi-tenant isolation
            if organization_id:
                org_uuid = uuid.UUID(organization_id)
                relation_query = relation_query.where(EntityRelation.organization_id == org_uuid)

            relation_query = relation_query.order_by(EntityRelation.weight.desc()).limit(max_neighbors)

            result = await self.db.execute(relation_query)
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
        organization_id: Optional[str] = None,
        access_tier_level: int = 100,
        user_id: Optional[str] = None,
        is_superadmin: bool = False,
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
            organization_id: Optional organization ID for multi-tenant isolation
            access_tier_level: Maximum access tier level for filtering
            user_id: Optional user ID for private document access
            is_superadmin: Whether the user is a superadmin

        Returns:
            GraphRAGContext with entities, relations, and chunks
        """
        # Step 1: Find query entities (filtered by organization)
        query_entities = await self.find_entities_by_query(
            query,
            limit=5,
            organization_id=organization_id,
        )

        if not query_entities:
            logger.debug("No entities found for query", query=query)
            return GraphRAGContext(
                entities=[],
                relations=[],
                chunks=[],
                graph_summary="No relevant entities found in knowledge graph.",
                entity_context="",
            )

        # Step 2: Expand through graph (filtered by organization)
        all_entities: Dict[uuid.UUID, Entity] = {}
        all_relations: List[EntityRelation] = []

        for entity in query_entities:
            all_entities[entity.id] = entity
            neighbors, relations = await self.get_entity_neighborhood(
                entity.id,
                max_hops=max_hops,
                max_neighbors=top_k,
                organization_id=organization_id,
            )
            for n in neighbors:
                all_entities[n.id] = n
            all_relations.extend(relations)

        # Step 3: Get document chunks for these entities with proper filtering
        entity_ids = list(all_entities.keys())

        # Build base query with joins for filtering
        chunk_query = (
            select(EntityMention)
            .options(selectinload(EntityMention.chunk).selectinload(Chunk.document))
            .join(Chunk, EntityMention.chunk_id == Chunk.id)
            .join(Document, Chunk.document_id == Document.id)
            .join(AccessTier, Document.access_tier_id == AccessTier.id)
            .where(EntityMention.entity_id.in_(entity_ids))
            # Filter by access tier
            .where(AccessTier.level <= access_tier_level)
        )

        # Filter by organization for multi-tenant isolation
        if organization_id:
            org_uuid = uuid.UUID(organization_id)
            chunk_query = chunk_query.where(Chunk.organization_id == org_uuid)

        # Filter private documents (only owner or superadmin can access)
        if not is_superadmin:
            if user_id:
                user_uuid = uuid.UUID(user_id)
                chunk_query = chunk_query.where(
                    or_(
                        Document.is_private == False,
                        and_(
                            Document.is_private == True,
                            Document.uploaded_by_id == user_uuid
                        )
                    )
                )
            else:
                chunk_query = chunk_query.where(Document.is_private == False)

        chunk_query = chunk_query.limit(top_k * 2)

        result = await self.db.execute(chunk_query)
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
