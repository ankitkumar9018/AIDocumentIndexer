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

import asyncio
import json
import re
import time
import uuid
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import structlog
from sqlalchemy import select, func, and_, or_, case, String
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


@dataclass
class MultiHopPath:
    """A single path through the knowledge graph."""
    entities: List[Entity]
    relations: List[EntityRelation]
    score: float  # Combined relevance score
    hop_count: int
    reasoning_steps: List[str] = field(default_factory=list)


@dataclass
class MultiHopResult:
    """Result of multi-hop reasoning query."""
    paths: List[MultiHopPath]
    entities_visited: List[Entity]
    total_hops: int
    reasoning_chain: str
    confidence: float  # Overall confidence in the multi-hop result
    query_entities: List[Entity]  # Seed entities from the query


# =============================================================================
# LLM Timeout Configuration and Helper
# =============================================================================

# Timeout configuration for LLM calls
LLM_BASE_TIMEOUT = 15  # Base timeout for network + model initialization
LLM_FIRST_TOKEN_TIMEOUT = 30  # Max wait for first token (stuck detection)
LLM_PROCESSING_TIME_PER_CHAR = 0.001  # ~0.001s per char (~4 chars per token)
LLM_GENERATION_TIME_PER_TOKEN = 0.1  # ~100ms per output token for local models
LLM_EXPECTED_OUTPUT_TOKENS = 1000  # Expected output size for entity extraction


def calculate_adaptive_timeout(text_length: int, expected_output_tokens: int = LLM_EXPECTED_OUTPUT_TOKENS) -> float:
    """
    Calculate timeout based on input text size.

    For a 10,000 char document (~2,500 tokens):
    - Processing: 10s
    - Generation: 100s (1000 tokens * 0.1s)
    - Base: 15s
    - Total: ~125s

    For a 50,000 char document (~12,500 tokens):
    - Processing: 50s
    - Generation: 100s
    - Base: 15s
    - Total: ~165s
    """
    processing_time = text_length * LLM_PROCESSING_TIME_PER_CHAR
    generation_time = expected_output_tokens * LLM_GENERATION_TIME_PER_TOKEN
    return LLM_BASE_TIMEOUT + processing_time + generation_time


async def call_llm_with_timeout(
    llm,
    prompt: str,
    text_length: int,
    invoke_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Call LLM with adaptive timeout based on text size.

    Uses asyncio.timeout for Python 3.11+ with fallback for older versions.
    Returns None on timeout instead of raising, for graceful handling.
    """
    timeout = calculate_adaptive_timeout(text_length)
    invoke_kwargs = invoke_kwargs or {}

    logger.debug(
        "LLM call starting with adaptive timeout",
        text_length=text_length,
        calculated_timeout=timeout,
    )

    start_time = time.time()

    try:
        # Use asyncio.timeout (Python 3.11+)
        async with asyncio.timeout(timeout):
            result = await llm.ainvoke(prompt, **invoke_kwargs)
            response = result.content if hasattr(result, 'content') else str(result)
            elapsed = time.time() - start_time
            logger.debug(
                "LLM call completed successfully",
                elapsed=elapsed,
                timeout=timeout,
                response_length=len(response) if response else 0,
            )
            return response
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(
            "LLM call timed out",
            timeout=timeout,
            text_length=text_length,
            elapsed=elapsed,
        )
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            "LLM call failed",
            error=str(e),
            elapsed=elapsed,
            text_length=text_length,
        )
        raise


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


# Simplified prompt for small models (<3B parameters)
ENTITY_EXTRACTION_PROMPT_SMALL = """Extract entities and relationships from this text. Output ONLY valid JSON.

Text:
{text}

ENTITY TYPES (pick one for each entity):
- PERSON: Names of people (e.g., "John Smith", "Dr. Jane Doe")
- ORGANIZATION: Companies, institutions (e.g., "Microsoft", "Harvard University")
- LOCATION: Places, cities, countries (e.g., "New York", "United States")
- CONCEPT: Abstract ideas (e.g., "Democracy", "Machine Learning")
- EVENT: Occurrences, meetings (e.g., "World War II", "Annual Meeting")
- PRODUCT: Products, services (e.g., "iPhone", "Azure Cloud")
- TECHNOLOGY: Technologies, tools (e.g., "Python", "Neural Networks")
- DATE: Dates, time periods (e.g., "2024", "January 1st")
- METRIC: Numbers, statistics (e.g., "50%", "$1.2M revenue")
- OTHER: Other important entities

RELATIONSHIP TYPES (if entities are related):
- WORKS_FOR: Person works for organization
- LOCATED_IN: Entity is located in a place
- PART_OF: Entity is part of another entity
- RELATED_TO: General relationship between entities
- CREATED_BY: Entity was created by another
- USES: Entity uses another entity

OUTPUT FORMAT (must be valid JSON):
{{
  "entities": [
    {{"name": "Microsoft", "type": "ORGANIZATION", "description": "Tech company", "aliases": ["MS"]}},
    {{"name": "Seattle", "type": "LOCATION", "description": "City in Washington", "aliases": []}}
  ],
  "relations": [
    {{"source": "Microsoft", "target": "Seattle", "type": "LOCATED_IN"}}
  ]
}}

IMPORTANT:
1. Output ONLY the JSON object, no other text
2. Use exact entity names from the text
3. Only include entities actually mentioned
4. "name" and "type" are required, others optional

JSON OUTPUT:"""


def _get_extraction_prompt_for_model(
    text: str,
    document_language: str,
    model_name: Optional[str] = None
) -> str:
    """
    Get entity extraction prompt optimized for model size.

    Small models (<3B) need simpler instructions with explicit JSON schema.
    Large models can handle detailed, nuanced prompts.

    Args:
        text: Text to extract from
        document_language: Human-readable language name
        model_name: Optional model name for optimization

    Returns:
        Formatted extraction prompt
    """
    # Check if this is a small model
    is_small = False
    if model_name:
        try:
            from backend.services.rag_module.prompts import is_tiny_model
            is_small = is_tiny_model(model_name)
        except ImportError:
            pass

    if is_small:
        # Use simplified prompt for small models
        return ENTITY_EXTRACTION_PROMPT_SMALL.format(text=text[:4000])  # Shorter context for small models
    else:
        # Use full prompt for large models
        return ENTITY_EXTRACTION_PROMPT.format(
            text=text[:8000],
            document_language=document_language,
        )


def _calculate_optimal_batch_size(model_name: Optional[str] = None) -> int:
    """
    Calculate optimal batch size for entity extraction based on model context window.

    Rules (based on typical model context windows and performance):
    - Tiny models (<3B): batch size 2 (smaller context ~8K tokens)
    - Small models (7B-13B): batch size 3 (standard context ~8-16K tokens)
    - Medium models (30B-70B): batch size 5 (larger context ~32K tokens)
    - Large models (>70B, 128K+ context): batch size 8 (huge context)

    Args:
        model_name: Optional model name for detection

    Returns:
        Optimal batch size (2-8)
    """
    if not model_name:
        return 3  # Safe default

    model_lower = model_name.lower()

    # Detect model size from name patterns
    # Tiny models (<3B)
    if any(size in model_lower for size in ["1b", "3b", "tiny", "mini"]):
        return 2

    # Small models (7B-13B)
    elif any(size in model_lower for size in ["7b", "8b", "9b", "13b"]):
        return 3

    # Medium models (30B-70B)
    elif any(size in model_lower for size in ["30b", "34b", "70b"]):
        return 5

    # Large models (identified by name, have huge context windows)
    elif any(model in model_lower for model in ["gpt-4", "gpt4", "claude-3", "claude-opus", "claude-sonnet", "gemini-pro", "gemini-ultra"]):
        return 8

    # Default for unknown models
    else:
        return 3


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
        use_fast_extraction: bool = True,  # Phase 2: Use dependency parsing for 80-90% cost savings
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        Extract entities and relationships from text using LLM with Phase 15 optimizations.

        Phase 2 Enhancement: Uses dependency parsing for simple text (94% accuracy)
        and falls back to LLM only for complex text, reducing costs by 80-90%.

        Args:
            text: Text to extract from
            document_id: Source document ID
            chunk_id: Source chunk ID
            document_language: Language code of the source document (e.g., "en", "de", "ru")
            use_fast_extraction: Use dependency parsing for simple text (default True)

        Returns:
            Tuple of (entities, relations)
        """
        # Phase 2: Try fast extraction first (80-90% cost reduction)
        if use_fast_extraction and document_language == "en":  # Fast extraction currently English-only
            try:
                from backend.services.dependency_entity_extractor import get_dependency_extractor

                fast_extractor = get_dependency_extractor()
                complexity = fast_extractor.estimate_complexity(text)

                if complexity < 0.7:  # Use fast extraction for simpler text
                    fast_entities, fast_relations = await fast_extractor.extract_entities(text)

                    # Convert to ExtractedEntity format
                    entities = []
                    for fe in fast_entities:
                        try:
                            # Map spaCy entity types to our EntityType enum
                            type_mapping = {
                                "PERSON": EntityType.PERSON,
                                "ORG": EntityType.ORGANIZATION,
                                "GPE": EntityType.LOCATION,
                                "LOC": EntityType.LOCATION,
                                "DATE": EntityType.DATE,
                                "MONEY": EntityType.NUMBER,
                                "PERCENT": EntityType.NUMBER,
                                "PRODUCT": EntityType.PRODUCT,
                                "EVENT": EntityType.EVENT,
                                "WORK_OF_ART": EntityType.CONCEPT,
                                "LAW": EntityType.CONCEPT,
                                "LANGUAGE": EntityType.CONCEPT,
                                "TECHNICAL": EntityType.CONCEPT,
                                "CONCEPT": EntityType.CONCEPT,
                                "ENTITY": EntityType.OTHER,
                            }
                            entity_type = type_mapping.get(fe.entity_type, EntityType.OTHER)

                            entities.append(ExtractedEntity(
                                name=fe.name,
                                entity_type=entity_type,
                                confidence=fe.confidence,
                                context=fe.context,
                                language=document_language,
                            ))
                        except Exception:
                            pass

                    # Convert relations
                    relations = []
                    for fr in fast_relations:
                        try:
                            relations.append(ExtractedRelation(
                                source_entity=fr.source_entity,
                                target_entity=fr.target_entity,
                                relation_type=RelationType.OTHER,
                                relation_label=fr.relation,
                                confidence=fr.confidence,
                            ))
                        except Exception:
                            pass

                    if entities:  # Only use fast extraction if we found something
                        logger.info(
                            "Used fast entity extraction (Phase 2 cost optimization)",
                            entity_count=len(entities),
                            relation_count=len(relations),
                            complexity=complexity,
                        )
                        return entities, relations

            except ImportError:
                logger.debug("Dependency extractor not available, using LLM")
            except Exception as e:
                logger.warning("Fast extraction failed, falling back to LLM", error=str(e))

        # Fall back to LLM extraction for complex text or non-English
        llm = await self._get_llm()

        # Graceful fallback: Return empty if LLM unavailable
        if not llm:
            logger.warning(
                "LLM not available for entity extraction - skipping",
                document_id=str(document_id) if document_id else None,
            )
            return [], []

        # Get model name for optimization
        model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)

        # Get human-readable language name for the prompt
        language_name = LANGUAGE_NAMES.get(document_language, "English")

        # Get model-specific prompt (simplified for small models)
        prompt = _get_extraction_prompt_for_model(
            text=text,
            document_language=language_name,
            model_name=model_name
        )

        try:
            # Apply Phase 15 optimizations for small models
            response = None
            is_small = False
            invoke_kwargs = {}

            if model_name:
                try:
                    from backend.services.rag_module.prompts import is_tiny_model, get_sampling_config
                    is_small = is_tiny_model(model_name)

                    if is_small:
                        # Use Phase 15 optimized parameters for small models
                        config = get_sampling_config(model_name)
                        invoke_kwargs = {
                            "temperature": config.get("temperature", 0.3),
                            "top_p": config.get("top_p"),
                            "top_k": config.get("top_k"),
                        }
                        # Remove None values
                        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}

                        logger.debug(
                            "Entity extraction using Phase 15 optimizations",
                            model=model_name,
                            temperature=config.get("temperature"),
                        )
                except ImportError:
                    pass

            # Use adaptive timeout for LLM call (prevents stuck on slow models)
            response = await call_llm_with_timeout(
                llm=llm,
                prompt=prompt,
                text_length=len(text),
                invoke_kwargs=invoke_kwargs if invoke_kwargs else None,
            )

            # Handle timeout (returns None)
            if response is None:
                logger.warning(
                    "LLM extraction timed out, skipping",
                    document_id=str(document_id) if document_id else None,
                    text_length=len(text),
                )
                return [], []

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
        batch_size: Optional[int] = None,
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        Extract entities from multiple text chunks in batches for better performance with adaptive batch sizing.

        Instead of making one LLM call per chunk, this method combines multiple
        chunks into single prompts, reducing LLM calls by ~3-5x.

        Batch size is automatically optimized based on model context window:
        - Tiny models (<3B): 2 chunks per batch
        - Small models (7B-13B): 3 chunks per batch
        - Medium models (30B-70B): 5 chunks per batch
        - Large models (GPT-4, Claude): 8 chunks per batch

        Args:
            chunks: List of text chunks to process
            document_id: Source document ID
            document_language: Language code of the source document
            batch_size: Optional batch size override (if None, calculates optimal size)

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

        # Calculate optimal batch size if not provided
        if batch_size is None:
            model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)
            batch_size = _calculate_optimal_batch_size(model_name)
            logger.info(
                "Using adaptive batch sizing for entity extraction",
                model=model_name,
                batch_size=batch_size,
                chunks=len(chunks),
            )

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
                # Use adaptive timeout for LLM call
                response = await call_llm_with_timeout(
                    llm=llm,
                    prompt=prompt,
                    text_length=len(combined_text),
                )

                if response is None:
                    logger.warning(
                        "Batch extraction timed out, skipping batch",
                        batch_index=i // batch_size,
                        text_length=len(combined_text),
                    )
                    continue

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
        # Use .first() instead of scalar_one_or_none() to handle duplicates gracefully
        # Order by created_at to consistently pick the oldest entity
        result = await self.db.execute(
            select(Entity).where(
                Entity.name_normalized == norm_name,
                Entity.entity_type == extracted.entity_type,
            ).order_by(Entity.created_at.asc()).limit(1)
        )
        entity = result.scalar_one_or_none()

        # Strategy 2: Try canonical name match (cross-language linking)
        # Use limit(1) to handle duplicates gracefully
        if not entity and canonical_norm != norm_name:
            result = await self.db.execute(
                select(Entity).where(
                    or_(
                        Entity.canonical_name == canonical,
                        Entity.name_normalized == canonical_norm,
                    ),
                    Entity.entity_type == extracted.entity_type,
                ).order_by(Entity.created_at.asc()).limit(1)
            )
            entity = result.scalar_one_or_none()

        # Strategy 3: Try ASCII-normalized match (cross-script)
        # Use limit(1) to handle duplicates gracefully
        if not entity:
            ascii_canonical = self._get_ascii_normalized(canonical)
            result = await self.db.execute(
                select(Entity).where(
                    func.lower(Entity.canonical_name) == ascii_canonical,
                    Entity.entity_type == extracted.entity_type,
                ).order_by(Entity.created_at.asc()).limit(1)
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
            # Create new entity with language support and embedding
            # Generate embedding inline for semantic search
            embedding = None
            try:
                from backend.services.embeddings import EmbeddingService
                import os

                # Use configured embedding provider (supports Ollama, HuggingFace, OpenAI)
                embedding_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
                embedding_model = None
                if embedding_provider.lower() == "ollama":
                    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", None)
                if not embedding_model:
                    embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", None)

                embedding_service = EmbeddingService(
                    provider=embedding_provider,
                    model=embedding_model
                )

                # Build entity text for embedding (name + description + aliases)
                text_parts = [extracted.name]
                if extracted.description:
                    text_parts.append(extracted.description)
                if extracted.aliases:
                    text_parts.extend(extracted.aliases[:3])  # Limit aliases
                entity_text = " ".join(text_parts)

                # Generate embedding
                embeddings = embedding_service.embed_texts([entity_text])
                if embeddings and len(embeddings) > 0:
                    embedding = embeddings[0]
            except Exception as e:
                logger.warning(
                    "Failed to generate entity embedding during creation",
                    entity_name=extracted.name,
                    error=str(e)
                )

            # Determine extraction method from source or default to "llm"
            extraction_method = getattr(extracted, 'source', 'llm') or 'llm'

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
                embedding=embedding,  # Store embedding
                confidence=extracted.confidence,  # Phase 2: Confidence scoring
                extraction_method=extraction_method,  # Phase 2: Track extraction method
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
        # Check if exists - use limit(1) to handle duplicates gracefully
        result = await self.db.execute(
            select(EntityRelation).where(
                EntityRelation.source_entity_id == source_entity_id,
                EntityRelation.target_entity_id == target_entity_id,
                EntityRelation.relation_type == relation_type,
            ).order_by(EntityRelation.created_at.asc()).limit(1)
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
            # Strategy 5: Query in aliases - use cast to text for database-agnostic search
            # This works for both PostgreSQL arrays and SQLite JSON arrays
            func.cast(Entity.aliases, String).ilike(f"%{query}%"),
        ]

        # Add language variant search if query_language specified
        if query_language:
            # Search in language_variants JSON field
            conditions.append(
                Entity.language_variants[query_language].astext.ilike(f"%{query}%")
            )

        base_query = select(Entity).where(or_(*conditions))

        # Filter by organization for multi-tenant isolation
        # PHASE 12 FIX: Include entities from user's org AND entities without org (legacy/shared)
        if organization_id:
            org_uuid = uuid.UUID(organization_id)
            base_query = base_query.where(
                or_(
                    Entity.organization_id == org_uuid,
                    Entity.organization_id.is_(None),  # Include entities without org
                )
            )

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

        # If not enough results, try embedding search (Phase 2 enhancement)
        if len(entities) < limit:
            try:
                # Get organization_id from the first entity if available
                org_id = None
                if entities:
                    org_id = str(entities[0].organization_id) if entities[0].organization_id else None

                # Search for semantically similar entities
                semantic_results = await self.search_entities_semantic(
                    query=query,
                    entity_types=entity_types,
                    top_k=limit - len(entities),
                    similarity_threshold=0.6,  # Lower threshold to get more candidates
                    organization_id=org_id,
                )

                # Add semantic results that aren't already in the list
                existing_ids = {e.id for e in entities}
                for entity, score in semantic_results:
                    if entity.id not in existing_ids:
                        entities.append(entity)
                        existing_ids.add(entity.id)
                        if len(entities) >= limit:
                            break

                if semantic_results:
                    logger.debug(
                        "Augmented entity search with semantic results",
                        text_results=len(entities) - len(semantic_results),
                        semantic_results=len(semantic_results),
                    )
            except Exception as e:
                logger.warning("Semantic entity search failed", error=str(e))

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
        # PHASE 12 FIX: Include entities from user's org AND entities without org (legacy/shared)
        if organization_id:
            org_uuid = uuid.UUID(organization_id)
            base_query = base_query.where(
                or_(
                    Entity.organization_id == org_uuid,
                    Entity.organization_id.is_(None),  # Include entities without org
                )
            )

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
        import os

        # Find entities without embeddings
        result = await self.db.execute(
            select(Entity).where(Entity.embedding.is_(None)).limit(batch_size)
        )
        entities = list(result.scalars().all())

        if not entities:
            logger.debug("All entities have embeddings")
            return 0

        # Use configured embedding provider (supports Ollama, HuggingFace, OpenAI)
        embedding_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        embedding_model = None
        if embedding_provider.lower() == "ollama":
            embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", None)
        if not embedding_model:
            embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", None)

        embedding_service = EmbeddingService(
            provider=embedding_provider,
            model=embedding_model
        )

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
            # PHASE 12 FIX: Include relations from user's org AND relations without org (legacy/shared)
            if organization_id:
                org_uuid = uuid.UUID(organization_id)
                relation_query = relation_query.where(
                    or_(
                        EntityRelation.organization_id == org_uuid,
                        EntityRelation.organization_id.is_(None),  # Include relations without org
                    )
                )

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
        # PHASE 12 FIX: Include chunks from user's org AND chunks without org (legacy/shared)
        if organization_id:
            org_uuid = uuid.UUID(organization_id)
            chunk_query = chunk_query.where(
                or_(
                    Chunk.organization_id == org_uuid,
                    Chunk.organization_id.is_(None),  # Include chunks without org
                )
            )

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
    # Multi-Hop Reasoning (Phase 2 Enhancement)
    # -------------------------------------------------------------------------

    async def multi_hop_query(
        self,
        query: str,
        max_hops: int = 3,
        beam_width: int = 10,
        strategy: str = "bfs_rf",
        relevance_threshold: float = 0.3,
    ) -> MultiHopResult:
        """
        Execute multi-hop reasoning over the knowledge graph.

        Uses configurable search strategies to find relevant paths through the graph,
        scoring each path by relevance to the original query.

        Research shows entity resolution improves multi-hop accuracy
        from 43% to 91% (Microsoft GraphRAG).

        Args:
            query: The user's question
            max_hops: Maximum traversal depth (default: 3)
            beam_width: Number of paths to keep at each hop (default: 10)
            strategy: Search strategy - "bfs_rf" (default), "beam_search", or "breadth_first"
                     - bfs_rf: BFS with Relevance Filtering (StepChain approach, recommended)
                     - beam_search: Traditional beam search
                     - breadth_first: Simple BFS exploration
            relevance_threshold: Minimum relevance score to continue path (0-1)

        Returns:
            MultiHopResult with paths, visited entities, and reasoning chain
        """
        logger.info(
            "Starting multi-hop reasoning",
            query=query[:100],
            max_hops=max_hops,
            strategy=strategy,
        )

        # 1. Identify seed entities from query using semantic search
        seed_entities = await self.search_entities_semantic(
            query=query,
            top_k=5,
            threshold=0.5,
        )

        if not seed_entities:
            # Fall back to keyword-based entity finding
            seed_entities = await self._extract_entities_from_query(query)

        if not seed_entities:
            logger.info("No seed entities found for multi-hop query")
            return MultiHopResult(
                paths=[],
                entities_visited=[],
                total_hops=0,
                reasoning_chain="No relevant entities found in the knowledge graph for this query.",
                confidence=0.0,
                query_entities=[],
            )

        logger.debug(
            "Found seed entities for multi-hop",
            seed_count=len(seed_entities),
            seed_names=[e.name for e in seed_entities[:5]],
        )

        # 2. Execute search through the graph based on strategy
        if strategy == "bfs_rf":
            # Use the enhanced BFS-RF (Breadth-First Search with Relevance Filtering)
            # Based on StepChain research showing +2.57 EM and +2.13 F1 improvement
            result = await self.bfs_rf_multi_hop(
                query=query,
                max_hops=max_hops,
                top_k_per_hop=beam_width,
                relevance_threshold=relevance_threshold,
            )
            # Return directly as BFS-RF already builds complete MultiHopResult
            return result
        elif strategy == "beam_search":
            paths = await self._beam_search_paths(
                seed_entities=seed_entities,
                query=query,
                max_hops=max_hops,
                beam_width=beam_width,
                relevance_threshold=relevance_threshold,
            )
        else:
            # Breadth-first as fallback
            paths = await self._breadth_first_paths(
                seed_entities=seed_entities,
                query=query,
                max_hops=max_hops,
                relevance_threshold=relevance_threshold,
            )

        # 3. Collect all visited entities
        visited_set: Set[uuid.UUID] = set()
        entities_visited: List[Entity] = []
        for path in paths:
            for entity in path.entities:
                if entity.id not in visited_set:
                    visited_set.add(entity.id)
                    entities_visited.append(entity)

        # 4. Build reasoning chain from top paths
        reasoning_chain = self._build_reasoning_chain(paths[:5], query)

        # 5. Calculate overall confidence
        if paths:
            confidence = sum(p.score for p in paths[:5]) / min(5, len(paths))
        else:
            confidence = 0.0

        total_hops = max(p.hop_count for p in paths) if paths else 0

        logger.info(
            "Multi-hop reasoning complete",
            paths_found=len(paths),
            entities_visited=len(entities_visited),
            total_hops=total_hops,
            confidence=confidence,
        )

        return MultiHopResult(
            paths=paths,
            entities_visited=entities_visited,
            total_hops=total_hops,
            reasoning_chain=reasoning_chain,
            confidence=confidence,
            query_entities=seed_entities,
        )

    async def _beam_search_paths(
        self,
        seed_entities: List[Entity],
        query: str,
        max_hops: int,
        beam_width: int,
        relevance_threshold: float,
    ) -> List[MultiHopPath]:
        """
        Beam search through the knowledge graph.

        Maintains top-k paths at each hop, scoring by relevance to query.
        """
        # Initialize frontier with seed entities
        # Each item: (current_entity, path_entities, path_relations, score, reasoning_steps)
        frontier: List[Tuple[Entity, List[Entity], List[EntityRelation], float, List[str]]] = [
            (entity, [entity], [], 1.0, [f"Start with '{entity.name}' ({entity.entity_type.value})"])
            for entity in seed_entities
        ]

        all_paths: List[MultiHopPath] = []
        visited_in_path: Dict[uuid.UUID, Set[uuid.UUID]] = {
            entity.id: {entity.id} for entity in seed_entities
        }

        for hop in range(max_hops):
            if not frontier:
                break

            new_frontier: List[Tuple[Entity, List[Entity], List[EntityRelation], float, List[str]]] = []

            for current_entity, path_entities, path_relations, path_score, reasoning_steps in frontier:
                # Get neighbors for current entity
                neighbors = await self._get_entity_neighbors(current_entity.id)

                for neighbor_entity, relation in neighbors:
                    # Skip if already in this path (avoid cycles)
                    path_key = path_entities[0].id
                    if path_key not in visited_in_path:
                        visited_in_path[path_key] = set()

                    if neighbor_entity.id in visited_in_path[path_key]:
                        continue

                    # Score relevance to query
                    relevance = await self._score_entity_relevance(neighbor_entity, query)

                    if relevance < relevance_threshold:
                        continue

                    # Calculate new path score (decay with distance)
                    hop_decay = 0.8 ** (hop + 1)
                    new_score = path_score * relevance * hop_decay

                    # Build reasoning step
                    step = f"'{current_entity.name}' --[{relation.relation_type.value}]--> '{neighbor_entity.name}'"
                    new_reasoning = reasoning_steps + [step]

                    # Add to new frontier
                    new_path_entities = path_entities + [neighbor_entity]
                    new_path_relations = path_relations + [relation]

                    new_frontier.append((
                        neighbor_entity,
                        new_path_entities,
                        new_path_relations,
                        new_score,
                        new_reasoning,
                    ))

                    # Mark as visited in this path
                    visited_in_path[path_key].add(neighbor_entity.id)

            # Keep top beam_width paths
            new_frontier.sort(key=lambda x: x[3], reverse=True)
            frontier = new_frontier[:beam_width]

            # Convert completed paths to MultiHopPath objects
            for _, path_entities, path_relations, score, reasoning_steps in frontier:
                all_paths.append(MultiHopPath(
                    entities=path_entities,
                    relations=path_relations,
                    score=score,
                    hop_count=hop + 1,
                    reasoning_steps=reasoning_steps,
                ))

        # Sort all paths by score and return top ones
        all_paths.sort(key=lambda p: p.score, reverse=True)
        return all_paths[:beam_width * 2]

    async def _breadth_first_paths(
        self,
        seed_entities: List[Entity],
        query: str,
        max_hops: int,
        relevance_threshold: float,
    ) -> List[MultiHopPath]:
        """
        Breadth-first search as simpler alternative to beam search.
        """
        paths: List[MultiHopPath] = []
        visited: Set[uuid.UUID] = set()

        # Queue: (entity, path_entities, path_relations, hop_count)
        queue: List[Tuple[Entity, List[Entity], List[EntityRelation], int]] = [
            (entity, [entity], [], 0) for entity in seed_entities
        ]

        while queue:
            current_entity, path_entities, path_relations, hop_count = queue.pop(0)

            if current_entity.id in visited:
                continue
            visited.add(current_entity.id)

            # Score and add path
            relevance = await self._score_entity_relevance(current_entity, query)
            if relevance >= relevance_threshold:
                paths.append(MultiHopPath(
                    entities=path_entities,
                    relations=path_relations,
                    score=relevance,
                    hop_count=hop_count,
                    reasoning_steps=[f"Visited '{e.name}'" for e in path_entities],
                ))

            # Continue if under max hops
            if hop_count < max_hops:
                neighbors = await self._get_entity_neighbors(current_entity.id)
                for neighbor_entity, relation in neighbors:
                    if neighbor_entity.id not in visited:
                        queue.append((
                            neighbor_entity,
                            path_entities + [neighbor_entity],
                            path_relations + [relation],
                            hop_count + 1,
                        ))

        paths.sort(key=lambda p: p.score, reverse=True)
        return paths[:20]

    async def bfs_rf_multi_hop(
        self,
        query: str,
        max_hops: int = 3,
        top_k_per_hop: int = 15,
        relevance_threshold: float = 0.25,
        diversity_penalty: float = 0.1,
        use_embedding_similarity: bool = True,
    ) -> MultiHopResult:
        """
        BFS-RF (Breadth-First Search with Relevance Filtering) for multi-hop reasoning.

        Implements the StepChain approach from research showing +2.57 EM and +2.13 F1
        improvement over standard GraphRAG.

        Key features:
        1. Dynamic graph maintenance during reasoning
        2. Relevance filtering at each hop to prune irrelevant paths
        3. Embedding-based similarity scoring for better relevance
        4. Path diversity to avoid redundant exploration
        5. Iterative context refinement

        Args:
            query: The user's question
            max_hops: Maximum traversal depth
            top_k_per_hop: Number of entities to keep at each hop
            relevance_threshold: Minimum relevance score to continue exploration
            diversity_penalty: Penalty for entities similar to already-visited ones
            use_embedding_similarity: Use embedding similarity for scoring

        Returns:
            MultiHopResult with enhanced reasoning paths
        """
        logger.info(
            "Starting BFS-RF multi-hop reasoning",
            query=query[:100],
            max_hops=max_hops,
            use_embeddings=use_embedding_similarity,
        )

        # 1. Get query embedding for similarity scoring
        query_embedding = None
        if use_embedding_similarity:
            try:
                embedding_service = get_embedding_service()
                query_embedding = embedding_service.embed_text(query)
            except Exception as e:
                logger.warning("Failed to get query embedding, falling back to keyword matching", error=str(e))

        # 2. Find seed entities using both semantic and keyword matching
        seed_entities = await self.search_entities_semantic(query=query, top_k=10, threshold=0.4)

        if len(seed_entities) < 3:
            # Supplement with keyword-based extraction
            keyword_entities = await self._extract_entities_from_query(query)
            seen_ids = {e.id for e in seed_entities}
            for e in keyword_entities:
                if e.id not in seen_ids:
                    seed_entities.append(e)
                    seen_ids.add(e.id)

        if not seed_entities:
            return MultiHopResult(
                paths=[],
                entities_visited=[],
                total_hops=0,
                reasoning_chain="No relevant entities found in knowledge graph.",
                confidence=0.0,
                query_entities=[],
            )

        logger.debug("BFS-RF seed entities", count=len(seed_entities), names=[e.name for e in seed_entities[:5]])

        # 3. Initialize BFS-RF state
        # Each frontier item: (entity, path, score, reasoning_context)
        frontier: List[Tuple[Entity, List[Entity], List[EntityRelation], float, str]] = [
            (e, [e], [], 1.0, f"Starting from '{e.name}' ({e.entity_type.value})")
            for e in seed_entities
        ]

        all_paths: List[MultiHopPath] = []
        visited_globally: Set[uuid.UUID] = set()
        entity_visit_count: Dict[uuid.UUID, int] = defaultdict(int)
        accumulated_context: List[str] = []

        # 4. BFS with Relevance Filtering
        for hop in range(max_hops):
            if not frontier:
                break

            logger.debug(f"BFS-RF hop {hop + 1}", frontier_size=len(frontier))

            new_frontier: List[Tuple[Entity, List[Entity], List[EntityRelation], float, str]] = []
            hop_entities_found: List[Entity] = []

            for current_entity, path_entities, path_relations, path_score, reasoning_ctx in frontier:
                # Get neighbors
                neighbors = await self._get_entity_neighbors(current_entity.id)

                for neighbor_entity, relation in neighbors:
                    # Skip already visited in this path
                    if neighbor_entity.id in {e.id for e in path_entities}:
                        continue

                    # Calculate relevance score
                    relevance = await self._score_entity_relevance_enhanced(
                        entity=neighbor_entity,
                        query=query,
                        query_embedding=query_embedding,
                        path_context=reasoning_ctx,
                        accumulated_context=accumulated_context,
                    )

                    # Apply diversity penalty for frequently visited entities
                    visit_penalty = diversity_penalty * entity_visit_count[neighbor_entity.id]
                    adjusted_relevance = max(0, relevance - visit_penalty)

                    # Filter by relevance threshold
                    if adjusted_relevance < relevance_threshold:
                        continue

                    # Calculate path score with hop decay
                    hop_decay = 0.85 ** (hop + 1)
                    new_score = path_score * adjusted_relevance * hop_decay

                    # Build reasoning context
                    relation_desc = relation.relation_type.value
                    if relation.description:
                        relation_desc = f"{relation_desc}: {relation.description[:50]}"

                    new_reasoning = (
                        f"{reasoning_ctx} → "
                        f"[{relation_desc}] → "
                        f"'{neighbor_entity.name}' ({neighbor_entity.entity_type.value})"
                    )

                    # Add to frontier
                    new_path = path_entities + [neighbor_entity]
                    new_relations = path_relations + [relation]

                    new_frontier.append((
                        neighbor_entity,
                        new_path,
                        new_relations,
                        new_score,
                        new_reasoning,
                    ))

                    hop_entities_found.append(neighbor_entity)
                    entity_visit_count[neighbor_entity.id] += 1

            # 5. Relevance Filtering: Keep top-k entities per hop
            new_frontier.sort(key=lambda x: x[3], reverse=True)
            frontier = new_frontier[:top_k_per_hop]

            # 6. Dynamic context accumulation
            if hop_entities_found:
                hop_summary = self._summarize_hop_findings(hop_entities_found, hop + 1)
                accumulated_context.append(hop_summary)

            # 7. Convert to paths
            for entity, path_entities, path_relations, score, reasoning in frontier:
                visited_globally.update(e.id for e in path_entities)

                # Create path with reasoning steps
                reasoning_steps = [
                    f"Hop {i+1}: {path_entities[i].name} → {path_entities[i+1].name}"
                    for i in range(len(path_entities) - 1)
                ]

                all_paths.append(MultiHopPath(
                    entities=path_entities,
                    relations=path_relations,
                    score=score,
                    hop_count=hop + 1,
                    reasoning_steps=reasoning_steps,
                ))

        # 8. Deduplicate and rank final paths
        unique_paths = self._deduplicate_paths(all_paths)
        unique_paths.sort(key=lambda p: (p.score, -p.hop_count), reverse=True)
        top_paths = unique_paths[:top_k_per_hop]

        # 9. Build reasoning chain
        reasoning_chain = self._build_enhanced_reasoning_chain(top_paths, query, accumulated_context)

        # 10. Collect visited entities
        entities_visited = []
        seen_ids = set()
        for path in top_paths:
            for entity in path.entities:
                if entity.id not in seen_ids:
                    entities_visited.append(entity)
                    seen_ids.add(entity.id)

        # 11. Calculate confidence
        if top_paths:
            # Weighted average of top path scores
            weights = [1.0 / (i + 1) for i in range(len(top_paths))]
            confidence = sum(p.score * w for p, w in zip(top_paths, weights)) / sum(weights)
        else:
            confidence = 0.0

        total_hops = max((p.hop_count for p in top_paths), default=0)

        logger.info(
            "BFS-RF multi-hop complete",
            paths_found=len(top_paths),
            entities_visited=len(entities_visited),
            total_hops=total_hops,
            confidence=round(confidence, 3),
        )

        return MultiHopResult(
            paths=top_paths,
            entities_visited=entities_visited,
            total_hops=total_hops,
            reasoning_chain=reasoning_chain,
            confidence=confidence,
            query_entities=seed_entities,
        )

    async def _score_entity_relevance_enhanced(
        self,
        entity: Entity,
        query: str,
        query_embedding: Optional[List[float]] = None,
        path_context: str = "",
        accumulated_context: List[str] = None,
    ) -> float:
        """
        Enhanced relevance scoring using multiple signals.

        Combines:
        1. Embedding similarity (if available)
        2. Keyword matching
        3. Entity type relevance
        4. Context coherence
        """
        score = 0.0
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # 1. Embedding similarity (primary signal when available)
        if query_embedding and entity.embedding:
            try:
                # Cosine similarity
                import numpy as np
                entity_emb = np.array(entity.embedding)
                query_emb = np.array(query_embedding)
                similarity = np.dot(entity_emb, query_emb) / (
                    np.linalg.norm(entity_emb) * np.linalg.norm(query_emb) + 1e-8
                )
                score += 0.5 * max(0, similarity)
            except Exception:
                pass

        # 2. Name matching
        entity_name_lower = entity.name.lower()
        if entity_name_lower in query_lower:
            score += 0.3
        else:
            entity_terms = set(entity_name_lower.split())
            overlap = len(query_terms & entity_terms)
            if overlap > 0:
                score += 0.2 * (overlap / max(len(query_terms), 1))

        # 3. Description relevance
        if entity.description:
            desc_lower = entity.description.lower()
            desc_terms = set(desc_lower.split())
            overlap = len(query_terms & desc_terms)
            if overlap > 0:
                score += 0.15 * (overlap / max(len(query_terms), 1))

        # 4. Type relevance
        type_boosts = {
            EntityType.PERSON: ["who", "person", "people", "author", "ceo", "founder", "leader"],
            EntityType.ORGANIZATION: ["company", "organization", "org", "team", "group", "business"],
            EntityType.LOCATION: ["where", "location", "place", "city", "country", "region"],
            EntityType.DATE: ["when", "date", "time", "year", "month", "day"],
            EntityType.TECHNOLOGY: ["how", "technology", "system", "software", "tool", "platform"],
            EntityType.CONCEPT: ["what", "concept", "idea", "theory", "method", "approach"],
            EntityType.EVENT: ["event", "conference", "meeting", "launch", "announcement"],
            EntityType.PRODUCT: ["product", "service", "feature", "release", "version"],
        }

        for etype, keywords in type_boosts.items():
            if entity.entity_type == etype and any(kw in query_lower for kw in keywords):
                score += 0.15
                break

        # 5. Context coherence (bonus for entities that fit the reasoning path)
        if path_context and entity.name.lower() in path_context.lower():
            score += 0.1

        if accumulated_context:
            context_text = " ".join(accumulated_context).lower()
            if entity.name.lower() in context_text:
                score += 0.05

        return min(1.0, score)

    def _summarize_hop_findings(self, entities: List[Entity], hop_num: int) -> str:
        """Create a brief summary of entities found at this hop."""
        if not entities:
            return ""

        # Group by type
        by_type: Dict[EntityType, List[str]] = defaultdict(list)
        for e in entities[:10]:  # Limit to top 10
            by_type[e.entity_type].append(e.name)

        parts = []
        for etype, names in by_type.items():
            names_str = ", ".join(names[:3])
            if len(names) > 3:
                names_str += f" (+{len(names) - 3} more)"
            parts.append(f"{etype.value}: {names_str}")

        return f"Hop {hop_num}: {'; '.join(parts)}"

    def _deduplicate_paths(self, paths: List[MultiHopPath]) -> List[MultiHopPath]:
        """Remove duplicate paths based on entity sequence."""
        seen_sequences = set()
        unique_paths = []

        for path in paths:
            # Create a hashable sequence identifier
            sequence = tuple(e.id for e in path.entities)
            if sequence not in seen_sequences:
                seen_sequences.add(sequence)
                unique_paths.append(path)

        return unique_paths

    def _build_enhanced_reasoning_chain(
        self,
        paths: List[MultiHopPath],
        query: str,
        accumulated_context: List[str],
    ) -> str:
        """Build a comprehensive reasoning chain from multi-hop results."""
        if not paths:
            return "No relevant reasoning paths found in the knowledge graph."

        parts = [f"Query: {query}\n"]

        # Add accumulated context summary
        if accumulated_context:
            parts.append("Discovery process:")
            for ctx in accumulated_context[:3]:
                parts.append(f"  - {ctx}")
            parts.append("")

        # Add top reasoning paths
        parts.append("Key reasoning paths:")
        for i, path in enumerate(paths[:5], 1):
            entity_chain = " → ".join(e.name for e in path.entities)
            parts.append(f"  {i}. {entity_chain} (score: {path.score:.2f}, hops: {path.hop_count})")

            # Add relation details
            if path.relations:
                for j, (e1, rel, e2) in enumerate(zip(path.entities[:-1], path.relations, path.entities[1:])):
                    parts.append(f"       {e1.name} --[{rel.relation_type.value}]--> {e2.name}")

        # Add entity summary
        all_entities = []
        seen = set()
        for path in paths:
            for e in path.entities:
                if e.id not in seen:
                    all_entities.append(e)
                    seen.add(e.id)

        if all_entities:
            parts.append(f"\nEntities discovered: {len(all_entities)}")
            by_type = defaultdict(list)
            for e in all_entities[:15]:
                by_type[e.entity_type.value].append(e.name)
            for etype, names in sorted(by_type.items()):
                parts.append(f"  - {etype}: {', '.join(names[:5])}")

        return "\n".join(parts)

    async def _get_entity_neighbors(
        self,
        entity_id: uuid.UUID,
    ) -> List[Tuple[Entity, EntityRelation]]:
        """Get all neighboring entities and their relations."""
        neighbors: List[Tuple[Entity, EntityRelation]] = []

        # Outgoing relations
        result = await self.db.execute(
            select(EntityRelation)
            .options(selectinload(EntityRelation.target_entity))
            .where(EntityRelation.source_entity_id == entity_id)
        )
        for relation in result.scalars():
            if relation.target_entity:
                neighbors.append((relation.target_entity, relation))

        # Incoming relations
        result = await self.db.execute(
            select(EntityRelation)
            .options(selectinload(EntityRelation.source_entity))
            .where(EntityRelation.target_entity_id == entity_id)
        )
        for relation in result.scalars():
            if relation.source_entity:
                neighbors.append((relation.source_entity, relation))

        return neighbors

    async def _score_entity_relevance(
        self,
        entity: Entity,
        query: str,
    ) -> float:
        """
        Score how relevant an entity is to the query.

        Uses a combination of:
        1. Name overlap with query terms
        2. Description relevance (if available)
        3. Entity type relevance
        """
        score = 0.0
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # Name matching
        entity_name_lower = entity.name.lower()
        if entity_name_lower in query_lower:
            score += 0.5
        else:
            # Partial term matching
            entity_terms = set(entity_name_lower.split())
            overlap = len(query_terms & entity_terms)
            if overlap > 0:
                score += 0.3 * (overlap / max(len(query_terms), 1))

        # Description relevance
        if entity.description:
            desc_lower = entity.description.lower()
            desc_terms = set(desc_lower.split())
            overlap = len(query_terms & desc_terms)
            if overlap > 0:
                score += 0.2 * (overlap / max(len(query_terms), 1))

        # Type relevance - certain types are more likely for certain query patterns
        type_boosts = {
            EntityType.PERSON: ["who", "person", "people", "author", "ceo", "founder"],
            EntityType.ORGANIZATION: ["company", "organization", "org", "team", "group"],
            EntityType.LOCATION: ["where", "location", "place", "city", "country"],
            EntityType.DATE: ["when", "date", "time", "year", "month"],
            EntityType.TECHNOLOGY: ["how", "technology", "system", "software", "tool"],
            EntityType.CONCEPT: ["what", "concept", "idea", "theory", "method"],
        }

        for etype, keywords in type_boosts.items():
            if entity.entity_type == etype and any(kw in query_lower for kw in keywords):
                score += 0.2
                break

        return min(1.0, score)

    async def _extract_entities_from_query(
        self,
        query: str,
    ) -> List[Entity]:
        """
        Extract entity names from query and find matching entities.

        Simple keyword extraction for fallback when semantic search fails.
        """
        # Extract capitalized words/phrases as potential entity names
        words = query.split()
        potential_entities = []

        for word in words:
            # Capitalized words (not at start of sentence)
            if word[0].isupper() and len(word) > 2:
                potential_entities.append(word.strip(".,!?"))

        # Also extract quoted phrases
        import re
        quoted = re.findall(r'"([^"]+)"', query) + re.findall(r"'([^']+)'", query)
        potential_entities.extend(quoted)

        if not potential_entities:
            return []

        # Search for these entities
        found_entities: List[Entity] = []
        for name in potential_entities[:5]:  # Limit to prevent too many queries
            result = await self.db.execute(
                select(Entity)
                .where(
                    or_(
                        func.lower(Entity.name) == name.lower(),
                        func.lower(Entity.name).contains(name.lower()),
                    )
                )
                .limit(2)
            )
            found_entities.extend(result.scalars())

        return found_entities[:5]

    def _build_reasoning_chain(
        self,
        paths: List[MultiHopPath],
        query: str,
    ) -> str:
        """Build a human-readable reasoning chain from paths."""
        if not paths:
            return "No reasoning paths found."

        lines = [f"Multi-hop reasoning for: '{query[:100]}'", ""]

        for i, path in enumerate(paths[:3]):
            lines.append(f"Path {i + 1} (confidence: {path.score:.2f}, {path.hop_count} hops):")
            for step in path.reasoning_steps:
                lines.append(f"  → {step}")
            lines.append("")

        # Summary
        all_entity_names = set()
        for path in paths:
            for entity in path.entities:
                all_entity_names.add(entity.name)

        lines.append(f"Entities discovered: {', '.join(list(all_entity_names)[:10])}")
        if len(all_entity_names) > 10:
            lines.append(f"  (and {len(all_entity_names) - 10} more)")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Entity Confidence Ranking (Phase 2 Enhancement)
    # -------------------------------------------------------------------------

    async def rank_entities_by_confidence(
        self,
        entities: List[Entity],
        query: Optional[str] = None,
        confidence_weight: float = 0.6,
        relevance_weight: float = 0.4,
    ) -> List[Tuple[Entity, float]]:
        """
        Rank entities by confidence and optional query relevance.

        Combines extraction confidence with query relevance for
        more accurate entity prioritization.

        Args:
            entities: List of entities to rank
            query: Optional query for relevance scoring
            confidence_weight: Weight for confidence (0-1)
            relevance_weight: Weight for relevance (0-1), should sum to 1

        Returns:
            List of (entity, combined_score) sorted by score descending
        """
        ranked: List[Tuple[Entity, float]] = []

        for entity in entities:
            # Base confidence (from extraction method)
            base_confidence = entity.confidence if entity.confidence else 0.5

            # Boost confidence based on mention count (more mentions = more reliable)
            mention_boost = min(0.2, (entity.mention_count - 1) * 0.02)
            adjusted_confidence = min(1.0, base_confidence + mention_boost)

            # Calculate query relevance if query provided
            if query:
                relevance = await self._score_entity_relevance(entity, query)
                combined_score = (
                    adjusted_confidence * confidence_weight +
                    relevance * relevance_weight
                )
            else:
                combined_score = adjusted_confidence

            ranked.append((entity, combined_score))

        # Sort by combined score descending
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    async def get_high_confidence_entities(
        self,
        min_confidence: float = 0.7,
        entity_type: Optional[EntityType] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """
        Get entities with high confidence scores.

        Args:
            min_confidence: Minimum confidence threshold (0-1)
            entity_type: Optional filter by entity type
            limit: Maximum entities to return

        Returns:
            List of high-confidence entities sorted by confidence
        """
        query = (
            select(Entity)
            .where(Entity.confidence >= min_confidence)
            .order_by(Entity.confidence.desc(), Entity.mention_count.desc())
            .limit(limit)
        )

        if entity_type:
            query = query.where(Entity.entity_type == entity_type)

        result = await self.db.execute(query)
        return list(result.scalars())

    async def get_entities_by_extraction_method(
        self,
        method: str,
        limit: int = 100,
    ) -> List[Entity]:
        """
        Get entities extracted by a specific method.

        Args:
            method: Extraction method ("llm", "dependency_parser", "ner", "pattern")
            limit: Maximum entities to return

        Returns:
            List of entities from the specified extraction method
        """
        result = await self.db.execute(
            select(Entity)
            .where(Entity.extraction_method == method)
            .order_by(Entity.confidence.desc())
            .limit(limit)
        )
        return list(result.scalars())

    def get_confidence_explanation(self, entity: Entity) -> str:
        """
        Generate human-readable explanation of entity confidence.

        Args:
            entity: Entity to explain

        Returns:
            Explanation string
        """
        method = entity.extraction_method or "unknown"
        confidence = entity.confidence if entity.confidence else 0.5
        mentions = entity.mention_count

        method_descriptions = {
            "llm": "Extracted by language model analysis",
            "dependency_parser": "Extracted by dependency parsing",
            "ner": "Extracted by named entity recognition",
            "pattern": "Extracted by pattern matching",
            "user_defined": "Defined by user",
        }

        method_desc = method_descriptions.get(method, f"Extracted by {method}")

        if confidence >= 0.9:
            conf_level = "very high"
        elif confidence >= 0.7:
            conf_level = "high"
        elif confidence >= 0.5:
            conf_level = "medium"
        else:
            conf_level = "low"

        return (
            f"{method_desc} with {conf_level} confidence ({confidence:.0%}). "
            f"Mentioned {mentions} time{'s' if mentions != 1 else ''} in documents."
        )

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
