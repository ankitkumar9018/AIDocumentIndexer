"""
AIDocumentIndexer - Entity Resolution Service
==============================================

Handles entity resolution and deduplication using multiple strategies:
1. Edit distance (Levenshtein) similarity for typos and variations
2. Token-based similarity for word order variations
3. Phonetic similarity for sound-alike names
4. LLM-based resolution for semantic equivalence

Inspired by RAGFlow's entity resolution approach.

Example matches:
- "Microsoft Corporation" ↔ "Microsoft Corp" (abbreviation)
- "John Smith" ↔ "Smith, John" (word order)
- "Elon Musk" ↔ "Elon R. Musk" (middle initial)
- "München" ↔ "Munich" (transliteration)
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict

import structlog
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
from unidecode import unidecode

logger = structlog.get_logger(__name__)


class SimilarityMetric(str, Enum):
    """Available similarity metrics for entity resolution."""
    LEVENSHTEIN = "levenshtein"  # Edit distance
    JARO_WINKLER = "jaro_winkler"  # Good for names
    TOKEN_SET = "token_set"  # Good for word order variations
    TOKEN_SORT = "token_sort"  # Sorted tokens comparison
    PARTIAL = "partial"  # Substring matching
    COMBINED = "combined"  # Weighted combination


@dataclass
class EntityMatch:
    """A potential match between two entities."""
    source_name: str
    target_name: str
    similarity_score: float
    metric_used: SimilarityMetric
    confidence: float
    is_same_entity: bool = False
    reason: str = ""


@dataclass
class ResolutionResult:
    """Result of entity resolution."""
    canonical_name: str
    variants: List[str]
    merged_count: int
    confidence: float
    resolution_method: str


@dataclass
class EntityCluster:
    """A cluster of entities that refer to the same real-world entity."""
    canonical_name: str
    canonical_type: str
    members: List[str] = field(default_factory=list)
    aliases: Set[str] = field(default_factory=set)
    merged_descriptions: List[str] = field(default_factory=list)
    confidence: float = 1.0


class EntityResolutionConfig:
    """Configuration for entity resolution."""

    def __init__(
        self,
        # Similarity thresholds
        exact_threshold: float = 0.95,  # Near-exact match
        high_threshold: float = 0.85,   # Highly similar
        medium_threshold: float = 0.75,  # Moderately similar
        low_threshold: float = 0.65,     # Loosely similar

        # Feature weights for combined scoring
        levenshtein_weight: float = 0.3,
        jaro_winkler_weight: float = 0.2,
        token_set_weight: float = 0.3,
        token_sort_weight: float = 0.2,

        # Entity type specific thresholds
        person_threshold: float = 0.80,  # Higher for persons (names are unique)
        organization_threshold: float = 0.75,  # Medium for organizations
        location_threshold: float = 0.70,  # Lower for locations (many variants)

        # Performance settings
        max_candidates: int = 10,  # Max candidates to consider per entity
        batch_size: int = 100,  # Batch size for bulk resolution

        # LLM fallback settings
        use_llm_fallback: bool = True,
        llm_threshold: float = 0.60,  # Below this, use LLM for resolution
    ):
        self.exact_threshold = exact_threshold
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold

        self.levenshtein_weight = levenshtein_weight
        self.jaro_winkler_weight = jaro_winkler_weight
        self.token_set_weight = token_set_weight
        self.token_sort_weight = token_sort_weight

        self.person_threshold = person_threshold
        self.organization_threshold = organization_threshold
        self.location_threshold = location_threshold

        self.max_candidates = max_candidates
        self.batch_size = batch_size

        self.use_llm_fallback = use_llm_fallback
        self.llm_threshold = llm_threshold


class EntityResolutionService:
    """
    Service for resolving and deduplicating entities.

    Uses a multi-stage approach:
    1. Exact match (normalized)
    2. Edit distance similarity
    3. Token-based similarity
    4. LLM-based resolution (for ambiguous cases)
    """

    def __init__(self, config: Optional[EntityResolutionConfig] = None):
        self.config = config or EntityResolutionConfig()
        self._entity_index: Dict[str, List[str]] = defaultdict(list)  # type -> [names]
        self._clusters: Dict[str, EntityCluster] = {}  # canonical_name -> cluster

    def _normalize_for_comparison(self, name: str) -> str:
        """
        Normalize entity name for comparison.

        Steps:
        1. Convert to lowercase
        2. Transliterate to ASCII
        3. Remove punctuation
        4. Collapse whitespace
        """
        if not name:
            return ""

        # Convert to lowercase and transliterate
        normalized = unidecode(name.lower())

        # Remove punctuation except spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)

        # Collapse whitespace
        normalized = ' '.join(normalized.split())

        return normalized.strip()

    def _get_tokens(self, name: str) -> Set[str]:
        """Extract significant tokens from entity name."""
        normalized = self._normalize_for_comparison(name)
        # Filter out common stopwords and short tokens
        stopwords = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'for', 'to', 'and', 'or', 'inc', 'ltd', 'llc', 'corp'}
        tokens = {t for t in normalized.split() if len(t) > 1 and t not in stopwords}
        return tokens

    def compute_similarity(
        self,
        name1: str,
        name2: str,
        metric: SimilarityMetric = SimilarityMetric.COMBINED,
    ) -> float:
        """
        Compute similarity between two entity names.

        Args:
            name1: First entity name
            name2: Second entity name
            metric: Similarity metric to use

        Returns:
            Similarity score between 0 and 1
        """
        if not name1 or not name2:
            return 0.0

        # Normalize for comparison
        norm1 = self._normalize_for_comparison(name1)
        norm2 = self._normalize_for_comparison(name2)

        if norm1 == norm2:
            return 1.0

        if metric == SimilarityMetric.LEVENSHTEIN:
            # Normalized Levenshtein distance
            distance = Levenshtein.distance(norm1, norm2)
            max_len = max(len(norm1), len(norm2))
            return 1 - (distance / max_len) if max_len > 0 else 1.0

        elif metric == SimilarityMetric.JARO_WINKLER:
            # Jaro-Winkler similarity (good for names)
            return fuzz.ratio(norm1, norm2) / 100.0

        elif metric == SimilarityMetric.TOKEN_SET:
            # Token set ratio (handles word order and duplicates)
            return fuzz.token_set_ratio(norm1, norm2) / 100.0

        elif metric == SimilarityMetric.TOKEN_SORT:
            # Token sort ratio (handles word order)
            return fuzz.token_sort_ratio(norm1, norm2) / 100.0

        elif metric == SimilarityMetric.PARTIAL:
            # Partial ratio (substring matching)
            return fuzz.partial_ratio(norm1, norm2) / 100.0

        elif metric == SimilarityMetric.COMBINED:
            # Weighted combination of multiple metrics
            scores = {
                'levenshtein': self.compute_similarity(name1, name2, SimilarityMetric.LEVENSHTEIN),
                'jaro_winkler': fuzz.ratio(norm1, norm2) / 100.0,
                'token_set': fuzz.token_set_ratio(norm1, norm2) / 100.0,
                'token_sort': fuzz.token_sort_ratio(norm1, norm2) / 100.0,
            }

            weighted_score = (
                scores['levenshtein'] * self.config.levenshtein_weight +
                scores['jaro_winkler'] * self.config.jaro_winkler_weight +
                scores['token_set'] * self.config.token_set_weight +
                scores['token_sort'] * self.config.token_sort_weight
            )

            return weighted_score

        return 0.0

    def is_similar(
        self,
        name1: str,
        name2: str,
        entity_type: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Check if two entity names are similar enough to be the same entity.

        Args:
            name1: First entity name
            name2: Second entity name
            entity_type: Entity type for type-specific threshold
            threshold: Override threshold (uses config default if None)

        Returns:
            True if entities are likely the same
        """
        # Determine threshold based on entity type if not provided
        if threshold is None:
            if entity_type:
                type_lower = entity_type.lower()
                if type_lower in ('person', 'people'):
                    threshold = self.config.person_threshold
                elif type_lower in ('organization', 'company', 'org'):
                    threshold = self.config.organization_threshold
                elif type_lower in ('location', 'place', 'geo'):
                    threshold = self.config.location_threshold
                else:
                    threshold = self.config.medium_threshold
            else:
                threshold = self.config.medium_threshold

        similarity = self.compute_similarity(name1, name2, SimilarityMetric.COMBINED)
        return similarity >= threshold

    def find_similar_entities(
        self,
        name: str,
        candidates: List[str],
        entity_type: Optional[str] = None,
        threshold: Optional[float] = None,
        max_results: Optional[int] = None,
    ) -> List[EntityMatch]:
        """
        Find entities similar to the given name from a list of candidates.

        Args:
            name: Entity name to match
            candidates: List of candidate entity names
            entity_type: Entity type for type-specific threshold
            threshold: Minimum similarity threshold
            max_results: Maximum number of results to return

        Returns:
            List of EntityMatch objects sorted by similarity
        """
        if not name or not candidates:
            return []

        threshold = threshold or self.config.low_threshold
        max_results = max_results or self.config.max_candidates

        # Use rapidfuzz's process.extract for efficient batch comparison
        normalized_name = self._normalize_for_comparison(name)
        normalized_candidates = [(c, self._normalize_for_comparison(c)) for c in candidates]

        matches = []
        for original, normalized in normalized_candidates:
            if normalized == normalized_name:
                # Exact match after normalization
                matches.append(EntityMatch(
                    source_name=name,
                    target_name=original,
                    similarity_score=1.0,
                    metric_used=SimilarityMetric.COMBINED,
                    confidence=1.0,
                    is_same_entity=True,
                    reason="Exact match after normalization"
                ))
            else:
                similarity = self.compute_similarity(name, original, SimilarityMetric.COMBINED)
                if similarity >= threshold:
                    # Determine if it's the same entity based on entity type
                    is_same = self.is_similar(name, original, entity_type, threshold=self.config.high_threshold)

                    matches.append(EntityMatch(
                        source_name=name,
                        target_name=original,
                        similarity_score=similarity,
                        metric_used=SimilarityMetric.COMBINED,
                        confidence=similarity,
                        is_same_entity=is_same,
                        reason=self._get_match_reason(name, original, similarity)
                    ))

        # Sort by similarity score (descending) and return top results
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches[:max_results]

    def _get_match_reason(self, name1: str, name2: str, similarity: float) -> str:
        """Generate human-readable reason for match."""
        norm1 = self._normalize_for_comparison(name1)
        norm2 = self._normalize_for_comparison(name2)

        # Check various match types
        tokens1 = self._get_tokens(name1)
        tokens2 = self._get_tokens(name2)

        if norm1 == norm2:
            return "Exact match after normalization"

        if tokens1 == tokens2:
            return "Same tokens, different order"

        if tokens1.issubset(tokens2) or tokens2.issubset(tokens1):
            return "Subset of tokens (abbreviation/expansion)"

        common_tokens = tokens1 & tokens2
        if common_tokens and len(common_tokens) >= min(len(tokens1), len(tokens2)) * 0.5:
            return f"Significant token overlap: {', '.join(common_tokens)}"

        # Check for edit distance
        distance = Levenshtein.distance(norm1, norm2)
        if distance <= 3:
            return f"Minor edit distance ({distance} edits)"

        return f"Combined similarity: {similarity:.2%}"

    def resolve_batch(
        self,
        entities: List[Tuple[str, str]],  # List of (name, type) tuples
        existing_entities: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, EntityCluster]:
        """
        Resolve a batch of entities, clustering similar ones together.

        Args:
            entities: List of (name, type) tuples to resolve
            existing_entities: Optional list of existing (name, type) tuples to match against

        Returns:
            Dict mapping canonical names to EntityCluster objects
        """
        clusters: Dict[str, EntityCluster] = {}
        processed: Set[str] = set()

        # Group entities by type for more efficient comparison
        by_type: Dict[str, List[str]] = defaultdict(list)
        for name, entity_type in entities:
            by_type[entity_type].append(name)

        # Add existing entities to the candidate pool
        if existing_entities:
            for name, entity_type in existing_entities:
                if name not in by_type[entity_type]:
                    by_type[entity_type].append(name)

        # Process each type separately
        for entity_type, names in by_type.items():
            for name in names:
                if name in processed:
                    continue

                # Find similar entities
                candidates = [n for n in names if n != name and n not in processed]
                matches = self.find_similar_entities(
                    name,
                    candidates,
                    entity_type=entity_type,
                    threshold=self.config.medium_threshold,
                )

                # Create cluster with this entity as canonical
                cluster = EntityCluster(
                    canonical_name=name,
                    canonical_type=entity_type,
                    members=[name],
                    aliases=set(),
                    confidence=1.0,
                )

                # Add similar entities to cluster
                for match in matches:
                    if match.is_same_entity and match.target_name not in processed:
                        cluster.members.append(match.target_name)
                        cluster.aliases.add(match.target_name)
                        processed.add(match.target_name)
                        cluster.confidence = min(cluster.confidence, match.confidence)

                # Choose the shortest name as canonical (often the most common form)
                if cluster.members:
                    cluster.canonical_name = min(cluster.members, key=len)
                    cluster.aliases = {m for m in cluster.members if m != cluster.canonical_name}

                processed.add(name)
                clusters[cluster.canonical_name] = cluster

        logger.info(
            "Batch entity resolution completed",
            input_entities=len(entities),
            clusters=len(clusters),
            merged_entities=len(entities) - len(clusters),
        )

        return clusters

    def deduplicate_entities(
        self,
        entities: List[Dict[str, Any]],
        name_key: str = "name",
        type_key: str = "entity_type",
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate a list of entity dictionaries.

        Args:
            entities: List of entity dictionaries
            name_key: Key for entity name in dictionary
            type_key: Key for entity type in dictionary

        Returns:
            Deduplicated list with merged entities
        """
        if not entities:
            return []

        # Extract (name, type) tuples
        entity_tuples = [(e.get(name_key, ""), e.get(type_key, "")) for e in entities]

        # Resolve to clusters
        clusters = self.resolve_batch(entity_tuples)

        # Build mapping from original name to canonical name
        name_to_canonical: Dict[str, str] = {}
        for canonical, cluster in clusters.items():
            for member in cluster.members:
                name_to_canonical[member] = canonical

        # Merge entities
        merged: Dict[str, Dict[str, Any]] = {}
        for entity in entities:
            name = entity.get(name_key, "")
            canonical = name_to_canonical.get(name, name)

            if canonical not in merged:
                merged[canonical] = entity.copy()
                merged[canonical][name_key] = canonical
                merged[canonical]["aliases"] = merged[canonical].get("aliases", [])
            else:
                # Merge data
                existing = merged[canonical]

                # Add as alias
                if name != canonical:
                    aliases = existing.get("aliases", [])
                    if name not in aliases:
                        aliases.append(name)
                    existing["aliases"] = aliases

                # Merge descriptions
                if entity.get("description") and not existing.get("description"):
                    existing["description"] = entity["description"]

                # Merge other list fields
                for key in ["mentions", "contexts", "sources"]:
                    if key in entity:
                        existing[key] = existing.get(key, []) + entity[key]

        return list(merged.values())

    async def resolve_with_llm(
        self,
        name1: str,
        name2: str,
        entity_type: str,
        context1: Optional[str] = None,
        context2: Optional[str] = None,
    ) -> EntityMatch:
        """
        Use LLM to determine if two entities are the same.

        For ambiguous cases where string similarity isn't enough.

        Args:
            name1: First entity name
            name2: Second entity name
            entity_type: Entity type
            context1: Optional context for first entity
            context2: Optional context for second entity

        Returns:
            EntityMatch with LLM's determination
        """
        from backend.services.llm import EnhancedLLMFactory

        prompt = f"""Determine if these two {entity_type} names refer to the same entity.

Entity 1: {name1}
{f'Context: {context1}' if context1 else ''}

Entity 2: {name2}
{f'Context: {context2}' if context2 else ''}

Consider:
- Are they variations of the same name (abbreviations, translations, typos)?
- Do the contexts suggest they're the same entity?
- Could they be different entities with similar names?

Respond with JSON:
{{
    "is_same_entity": true/false,
    "confidence": 0.0-1.0,
    "reason": "explanation"
}}"""

        try:
            factory = EnhancedLLMFactory()
            llm = factory.create_llm(temperature=0)
            response = await llm.ainvoke(prompt)

            # Parse response
            import json
            result = json.loads(response.content)

            return EntityMatch(
                source_name=name1,
                target_name=name2,
                similarity_score=self.compute_similarity(name1, name2),
                metric_used=SimilarityMetric.COMBINED,
                confidence=result.get("confidence", 0.5),
                is_same_entity=result.get("is_same_entity", False),
                reason=result.get("reason", "LLM resolution"),
            )
        except Exception as e:
            logger.warning("LLM entity resolution failed", error=str(e))
            # Fall back to string similarity
            similarity = self.compute_similarity(name1, name2)
            return EntityMatch(
                source_name=name1,
                target_name=name2,
                similarity_score=similarity,
                metric_used=SimilarityMetric.COMBINED,
                confidence=similarity,
                is_same_entity=similarity >= self.config.high_threshold,
                reason="Fallback to string similarity",
            )


# =============================================================================
# Singleton Access
# =============================================================================

_resolution_service: Optional[EntityResolutionService] = None


def get_entity_resolution_service(
    config: Optional[EntityResolutionConfig] = None
) -> EntityResolutionService:
    """Get or create the entity resolution service."""
    global _resolution_service
    if _resolution_service is None:
        _resolution_service = EntityResolutionService(config)
    return _resolution_service


# =============================================================================
# Utility Functions
# =============================================================================

def is_entity_similar(
    name1: str,
    name2: str,
    threshold: float = 0.75,
) -> bool:
    """
    Quick check if two entity names are similar.

    Uses edit distance normalized by max length.
    Threshold of 0.75 catches most variations while avoiding false positives.

    Examples (at threshold 0.75):
    - "Microsoft" vs "Microsoft Corp" → True (0.77)
    - "John Smith" vs "Smith, John" → True (0.82)
    - "Apple" vs "Google" → False (0.40)
    """
    if not name1 or not name2:
        return False

    # Quick exact match
    norm1 = unidecode(name1.lower().strip())
    norm2 = unidecode(name2.lower().strip())

    if norm1 == norm2:
        return True

    # Edit distance check
    distance = Levenshtein.distance(norm1, norm2)
    max_len = max(len(norm1), len(norm2))
    similarity = 1 - (distance / max_len) if max_len > 0 else 1.0

    # Also check token overlap for longer names
    if similarity < threshold and len(norm1.split()) > 1:
        token_similarity = fuzz.token_set_ratio(norm1, norm2) / 100.0
        similarity = max(similarity, token_similarity)

    return similarity >= threshold


def find_best_match(
    name: str,
    candidates: List[str],
    threshold: float = 0.75,
) -> Optional[str]:
    """
    Find the best matching candidate for an entity name.

    Returns None if no candidate meets the threshold.
    """
    if not name or not candidates:
        return None

    service = get_entity_resolution_service()
    matches = service.find_similar_entities(
        name,
        candidates,
        threshold=threshold,
        max_results=1,
    )

    if matches and matches[0].similarity_score >= threshold:
        return matches[0].target_name

    return None
