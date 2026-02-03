"""
AIDocumentIndexer - Smart Collections Service
==============================================

Automatically organizes documents into intelligent collections using:
- Topic clustering (K-means, hierarchical)
- Semantic similarity grouping
- Content-based tagging
- Time-based organization
- Entity-based grouping (people, companies, projects)
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import json

import structlog
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger(__name__)


class OrganizationStrategy(str, Enum):
    """Strategies for organizing documents."""
    TOPIC_CLUSTER = "topic_cluster"  # Group by semantic topics
    ENTITY_BASED = "entity_based"  # Group by extracted entities
    TIME_BASED = "time_based"  # Group by time periods
    SIMILARITY = "similarity"  # Group by content similarity
    HYBRID = "hybrid"  # Combine multiple strategies


class CollectionType(str, Enum):
    """Types of smart collections."""
    AUTO_TOPIC = "auto_topic"  # AI-generated topic clusters
    AUTO_PROJECT = "auto_project"  # Detected project groupings
    AUTO_PERSON = "auto_person"  # Documents about specific people
    AUTO_COMPANY = "auto_company"  # Documents about companies
    AUTO_TIME = "auto_time"  # Time-based groupings
    AUTO_SIMILAR = "auto_similar"  # Similarity-based groupings
    USER_DEFINED = "user_defined"  # User-created collections


@dataclass
class SmartCollection:
    """Represents an auto-generated collection."""
    id: str
    name: str
    description: str
    collection_type: CollectionType
    document_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    confidence_score: float = 0.0
    keywords: List[str] = field(default_factory=list)
    representative_doc_id: Optional[str] = None
    parent_collection_id: Optional[str] = None
    child_collection_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "collection_type": self.collection_type.value,
            "document_ids": self.document_ids,
            "document_count": len(self.document_ids),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confidence_score": self.confidence_score,
            "keywords": self.keywords,
            "representative_doc_id": self.representative_doc_id,
            "parent_collection_id": self.parent_collection_id,
            "child_collection_ids": self.child_collection_ids,
        }


@dataclass
class OrganizationResult:
    """Result of document organization."""
    collections: List[SmartCollection]
    uncategorized_docs: List[str]
    strategy_used: OrganizationStrategy
    processing_time_ms: float
    total_documents: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collections": [c.to_dict() for c in self.collections],
            "uncategorized_docs": self.uncategorized_docs,
            "strategy_used": self.strategy_used.value,
            "processing_time_ms": self.processing_time_ms,
            "total_documents": self.total_documents,
            "collection_count": len(self.collections),
        }


class SmartCollectionsService:
    """
    Service for automatically organizing documents into smart collections.

    Features:
    - Topic clustering using embeddings
    - Entity-based grouping (people, companies, projects)
    - Time-based organization
    - Hierarchical collection structure
    - Incremental updates
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        max_clusters: int = 20,
        similarity_threshold: float = 0.75,
        llm_client: Optional[Any] = None,
    ):
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.similarity_threshold = similarity_threshold
        self.llm_client = llm_client
        self._collection_cache: Dict[str, SmartCollection] = {}

    async def organize_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
        strategy: OrganizationStrategy = OrganizationStrategy.HYBRID,
        existing_collections: Optional[List[SmartCollection]] = None,
    ) -> OrganizationResult:
        """
        Organize documents into smart collections.

        Args:
            documents: List of document dicts with id, title, content, metadata
            embeddings: Document embeddings as numpy array
            strategy: Organization strategy to use
            existing_collections: Existing collections to update

        Returns:
            OrganizationResult with generated collections
        """
        start_time = datetime.utcnow()

        if len(documents) < self.min_cluster_size:
            return OrganizationResult(
                collections=[],
                uncategorized_docs=[d["id"] for d in documents],
                strategy_used=strategy,
                processing_time_ms=0,
                total_documents=len(documents),
            )

        collections: List[SmartCollection] = []
        uncategorized: Set[str] = set(d["id"] for d in documents)

        if strategy == OrganizationStrategy.TOPIC_CLUSTER:
            topic_collections = await self._cluster_by_topic(documents, embeddings)
            collections.extend(topic_collections)

        elif strategy == OrganizationStrategy.ENTITY_BASED:
            entity_collections = await self._group_by_entities(documents)
            collections.extend(entity_collections)

        elif strategy == OrganizationStrategy.TIME_BASED:
            time_collections = await self._group_by_time(documents)
            collections.extend(time_collections)

        elif strategy == OrganizationStrategy.SIMILARITY:
            similar_collections = await self._group_by_similarity(documents, embeddings)
            collections.extend(similar_collections)

        elif strategy == OrganizationStrategy.HYBRID:
            # Combine multiple strategies
            topic_collections = await self._cluster_by_topic(documents, embeddings)
            entity_collections = await self._group_by_entities(documents)
            time_collections = await self._group_by_time(documents)

            # Merge overlapping collections
            collections = await self._merge_collections(
                topic_collections + entity_collections + time_collections
            )

        # Update uncategorized set
        for collection in collections:
            uncategorized -= set(collection.document_ids)

        # Generate names and descriptions using LLM
        if self.llm_client:
            collections = await self._enhance_collections_with_llm(collections, documents)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return OrganizationResult(
            collections=collections,
            uncategorized_docs=list(uncategorized),
            strategy_used=strategy,
            processing_time_ms=processing_time,
            total_documents=len(documents),
        )

    async def _cluster_by_topic(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> List[SmartCollection]:
        """Cluster documents by topic using K-means on embeddings."""
        n_docs = len(documents)

        # Determine optimal number of clusters
        n_clusters = min(
            self.max_clusters,
            max(2, n_docs // self.min_cluster_size)
        )

        # Apply K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        labels = kmeans.fit_predict(embeddings)

        # Group documents by cluster
        clusters: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for doc, label in zip(documents, labels):
            clusters[label].append(doc)

        collections = []
        for cluster_id, cluster_docs in clusters.items():
            if len(cluster_docs) < self.min_cluster_size:
                continue

            # Find representative document (closest to centroid)
            cluster_embeddings = embeddings[[
                i for i, l in enumerate(labels) if l == cluster_id
            ]]
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            rep_idx = np.argmin(distances)
            representative_doc = cluster_docs[rep_idx]

            # Extract keywords from cluster
            keywords = await self._extract_cluster_keywords(cluster_docs)

            collection = SmartCollection(
                id=self._generate_collection_id("topic", cluster_id),
                name=f"Topic {cluster_id + 1}",  # Will be enhanced by LLM
                description="Auto-generated topic cluster",
                collection_type=CollectionType.AUTO_TOPIC,
                document_ids=[d["id"] for d in cluster_docs],
                confidence_score=self._calculate_cluster_cohesion(cluster_embeddings),
                keywords=keywords,
                representative_doc_id=representative_doc["id"],
            )
            collections.append(collection)

        return collections

    async def _group_by_entities(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[SmartCollection]:
        """Group documents by extracted entities (people, companies, projects)."""
        entity_docs: Dict[str, Dict[str, List[str]]] = {
            "person": defaultdict(list),
            "company": defaultdict(list),
            "project": defaultdict(list),
        }

        for doc in documents:
            entities = doc.get("metadata", {}).get("entities", {})

            for person in entities.get("people", []):
                entity_docs["person"][person.lower()].append(doc["id"])

            for company in entities.get("companies", []):
                entity_docs["company"][company.lower()].append(doc["id"])

            for project in entities.get("projects", []):
                entity_docs["project"][project.lower()].append(doc["id"])

        collections = []

        # Create collections for entities with enough documents
        entity_type_map = {
            "person": CollectionType.AUTO_PERSON,
            "company": CollectionType.AUTO_COMPANY,
            "project": CollectionType.AUTO_PROJECT,
        }

        for entity_type, entity_dict in entity_docs.items():
            for entity_name, doc_ids in entity_dict.items():
                if len(doc_ids) >= self.min_cluster_size:
                    collection = SmartCollection(
                        id=self._generate_collection_id(entity_type, entity_name),
                        name=entity_name.title(),
                        description=f"Documents related to {entity_name}",
                        collection_type=entity_type_map[entity_type],
                        document_ids=list(set(doc_ids)),
                        confidence_score=0.9,  # High confidence for entity matches
                        keywords=[entity_name],
                        metadata={"entity_type": entity_type, "entity_name": entity_name},
                    )
                    collections.append(collection)

        return collections

    async def _group_by_time(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[SmartCollection]:
        """Group documents by time periods."""
        time_buckets: Dict[str, List[str]] = defaultdict(list)

        for doc in documents:
            created_at = doc.get("created_at") or doc.get("metadata", {}).get("created_at")
            if not created_at:
                continue

            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    continue

            # Bucket by month
            bucket_key = created_at.strftime("%Y-%m")
            time_buckets[bucket_key].append(doc["id"])

        collections = []
        for bucket_key, doc_ids in sorted(time_buckets.items(), reverse=True):
            if len(doc_ids) >= self.min_cluster_size:
                year, month = bucket_key.split("-")
                month_name = datetime(int(year), int(month), 1).strftime("%B %Y")

                collection = SmartCollection(
                    id=self._generate_collection_id("time", bucket_key),
                    name=month_name,
                    description=f"Documents from {month_name}",
                    collection_type=CollectionType.AUTO_TIME,
                    document_ids=doc_ids,
                    confidence_score=1.0,  # Time grouping is deterministic
                    metadata={"year": year, "month": month},
                )
                collections.append(collection)

        return collections

    async def _group_by_similarity(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> List[SmartCollection]:
        """Group documents by content similarity using hierarchical clustering."""
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Apply hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - self.similarity_threshold,
            metric="precomputed",
            linkage="average",
        )

        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        labels = clustering.fit_predict(distance_matrix)

        # Group documents by cluster
        clusters: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for doc, label in zip(documents, labels):
            clusters[label].append(doc)

        collections = []
        for cluster_id, cluster_docs in clusters.items():
            if len(cluster_docs) < self.min_cluster_size:
                continue

            # Calculate average similarity within cluster
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            cluster_sim = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
            avg_similarity = np.mean(cluster_sim[np.triu_indices_from(cluster_sim, k=1)])

            collection = SmartCollection(
                id=self._generate_collection_id("similar", cluster_id),
                name=f"Similar Documents {cluster_id + 1}",
                description="Documents with similar content",
                collection_type=CollectionType.AUTO_SIMILAR,
                document_ids=[d["id"] for d in cluster_docs],
                confidence_score=float(avg_similarity),
                metadata={"avg_similarity": float(avg_similarity)},
            )
            collections.append(collection)

        return collections

    async def _merge_collections(
        self,
        collections: List[SmartCollection],
    ) -> List[SmartCollection]:
        """Merge overlapping collections from different strategies."""
        if not collections:
            return []

        # Calculate overlap between collections
        merged = []
        used = set()

        for i, c1 in enumerate(collections):
            if i in used:
                continue

            docs1 = set(c1.document_ids)
            to_merge = [c1]

            for j, c2 in enumerate(collections[i + 1:], start=i + 1):
                if j in used:
                    continue

                docs2 = set(c2.document_ids)
                overlap = len(docs1 & docs2) / min(len(docs1), len(docs2))

                # Merge if >50% overlap
                if overlap > 0.5:
                    to_merge.append(c2)
                    used.add(j)
                    docs1 |= docs2

            if len(to_merge) == 1:
                merged.append(c1)
            else:
                # Create merged collection
                all_docs = list(set(
                    doc_id
                    for c in to_merge
                    for doc_id in c.document_ids
                ))
                all_keywords = list(set(
                    kw
                    for c in to_merge
                    for kw in c.keywords
                ))[:10]

                merged_collection = SmartCollection(
                    id=self._generate_collection_id("merged", i),
                    name=to_merge[0].name,  # Use name from highest confidence
                    description="Merged collection from multiple strategies",
                    collection_type=to_merge[0].collection_type,
                    document_ids=all_docs,
                    confidence_score=max(c.confidence_score for c in to_merge),
                    keywords=all_keywords,
                    metadata={
                        "merged_from": [c.collection_type.value for c in to_merge],
                        "merge_count": len(to_merge),
                    },
                )
                merged.append(merged_collection)

            used.add(i)

        return merged

    async def _enhance_collections_with_llm(
        self,
        collections: List[SmartCollection],
        documents: List[Dict[str, Any]],
    ) -> List[SmartCollection]:
        """Use LLM to generate better names and descriptions."""
        if not self.llm_client:
            return collections

        doc_lookup = {d["id"]: d for d in documents}

        for collection in collections:
            # Get sample documents for context
            sample_docs = [
                doc_lookup.get(doc_id, {})
                for doc_id in collection.document_ids[:5]
            ]
            sample_titles = [d.get("title", "Untitled") for d in sample_docs]
            sample_snippets = [
                d.get("content", "")[:200]
                for d in sample_docs
            ]

            try:
                prompt = f"""Given these document titles and snippets from a collection, generate:
1. A concise, descriptive name (3-5 words)
2. A brief description (1-2 sentences)

Titles: {sample_titles}
Snippets: {sample_snippets}
Keywords: {collection.keywords}

Respond in JSON format:
{{"name": "...", "description": "..."}}"""

                response = await self.llm_client.complete(prompt)
                result = json.loads(response)

                collection.name = result.get("name", collection.name)
                collection.description = result.get("description", collection.description)

            except Exception as e:
                logger.warning(
                    "Failed to enhance collection with LLM",
                    collection_id=collection.id,
                    error=str(e),
                )

        return collections

    async def _extract_cluster_keywords(
        self,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[str]:
        """Extract representative keywords from a cluster of documents."""
        from collections import Counter
        import re

        # Simple keyword extraction using word frequency
        word_counts: Counter = Counter()
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "this",
            "that", "these", "those", "it", "its", "as", "if", "when", "where",
            "which", "who", "whom", "what", "how", "why", "all", "each", "every",
            "both", "few", "more", "most", "other", "some", "such", "no", "not",
            "only", "same", "so", "than", "too", "very", "just", "also", "now",
        }

        for doc in documents:
            content = doc.get("content", "") + " " + doc.get("title", "")
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            for word in words:
                if word not in stop_words:
                    word_counts[word] += 1

        return [word for word, _ in word_counts.most_common(top_k)]

    def _calculate_cluster_cohesion(self, embeddings: np.ndarray) -> float:
        """Calculate cohesion score for a cluster (average pairwise similarity)."""
        if len(embeddings) < 2:
            return 1.0

        similarities = cosine_similarity(embeddings)
        # Get upper triangle (excluding diagonal)
        upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
        return float(np.mean(upper_tri))

    def _generate_collection_id(self, prefix: str, identifier: Any) -> str:
        """Generate a unique collection ID."""
        content = f"{prefix}_{identifier}_{datetime.utcnow().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def suggest_collection_for_document(
        self,
        document: Dict[str, Any],
        embedding: np.ndarray,
        existing_collections: List[SmartCollection],
        collection_embeddings: Dict[str, np.ndarray],
    ) -> List[Tuple[SmartCollection, float]]:
        """
        Suggest existing collections for a new document.

        Returns list of (collection, similarity_score) tuples.
        """
        suggestions = []

        for collection in existing_collections:
            if collection.id not in collection_embeddings:
                continue

            coll_embedding = collection_embeddings[collection.id]
            similarity = float(cosine_similarity(
                embedding.reshape(1, -1),
                coll_embedding.reshape(1, -1)
            )[0, 0])

            if similarity >= self.similarity_threshold:
                suggestions.append((collection, similarity))

        # Sort by similarity
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:5]  # Top 5 suggestions

    async def update_collection_incrementally(
        self,
        collection: SmartCollection,
        new_documents: List[Dict[str, Any]],
        new_embeddings: np.ndarray,
        existing_embeddings: np.ndarray,
    ) -> SmartCollection:
        """
        Incrementally update a collection with new documents.

        Only adds documents that meet similarity threshold.
        """
        if len(new_documents) == 0:
            return collection

        # Calculate centroid of existing collection
        centroid = np.mean(existing_embeddings, axis=0)

        # Check each new document
        for doc, embedding in zip(new_documents, new_embeddings):
            similarity = float(cosine_similarity(
                embedding.reshape(1, -1),
                centroid.reshape(1, -1)
            )[0, 0])

            if similarity >= self.similarity_threshold:
                if doc["id"] not in collection.document_ids:
                    collection.document_ids.append(doc["id"])

        collection.updated_at = datetime.utcnow()
        return collection


# Singleton instance
_smart_collections_service: Optional[SmartCollectionsService] = None


def get_smart_collections_service() -> SmartCollectionsService:
    """Get or create the smart collections service singleton."""
    global _smart_collections_service
    if _smart_collections_service is None:
        _smart_collections_service = SmartCollectionsService()
    return _smart_collections_service
