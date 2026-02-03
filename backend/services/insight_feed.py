"""
AIDocumentIndexer - Insight Feed Service
==========================================

Proactive intelligence system that:
- Surfaces relevant documents based on user activity
- Detects trends and patterns in document corpus
- Generates daily/weekly digests
- Identifies knowledge gaps
- Suggests connections between documents
- Alerts on document updates and changes
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
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger(__name__)


class InsightType(str, Enum):
    """Types of insights the system can generate."""
    TRENDING_TOPIC = "trending_topic"  # Topics gaining attention
    KNOWLEDGE_GAP = "knowledge_gap"  # Areas with sparse coverage
    DOCUMENT_CONNECTION = "document_connection"  # Related documents
    ACTIVITY_PATTERN = "activity_pattern"  # User behavior patterns
    CONTENT_UPDATE = "content_update"  # Document changes
    STALE_CONTENT = "stale_content"  # Outdated documents
    DUPLICATE_DETECTION = "duplicate_detection"  # Similar documents
    EXPERT_SUGGESTION = "expert_suggestion"  # Recommended experts
    READING_RECOMMENDATION = "reading_recommendation"  # Personalized suggestions
    SUMMARY_DIGEST = "summary_digest"  # Periodic summaries


class InsightPriority(str, Enum):
    """Priority levels for insights."""
    CRITICAL = "critical"  # Requires immediate attention
    HIGH = "high"  # Important but not urgent
    MEDIUM = "medium"  # Useful information
    LOW = "low"  # Nice to know


@dataclass
class Insight:
    """Represents a single insight."""
    id: str
    insight_type: InsightType
    title: str
    description: str
    priority: InsightPriority
    relevance_score: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_document_ids: List[str] = field(default_factory=list)
    related_entity_ids: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    is_read: bool = False
    is_dismissed: bool = False
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "insight_type": self.insight_type.value,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "relevance_score": self.relevance_score,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
            "related_document_ids": self.related_document_ids,
            "related_entity_ids": self.related_entity_ids,
            "action_items": self.action_items,
            "is_read": self.is_read,
            "is_dismissed": self.is_dismissed,
        }


@dataclass
class UserActivityProfile:
    """Tracks user activity for personalized insights."""
    user_id: str
    viewed_documents: List[Tuple[str, datetime]] = field(default_factory=list)
    searched_queries: List[Tuple[str, datetime]] = field(default_factory=list)
    created_documents: List[Tuple[str, datetime]] = field(default_factory=list)
    frequent_topics: Dict[str, int] = field(default_factory=dict)
    active_hours: Dict[int, int] = field(default_factory=dict)
    last_active: Optional[datetime] = None


@dataclass
class InsightFeedConfig:
    """Configuration for insight generation."""
    trending_window_days: int = 7
    stale_threshold_days: int = 90
    similarity_threshold: float = 0.85
    max_insights_per_type: int = 5
    digest_frequency_hours: int = 24
    min_documents_for_trend: int = 3
    knowledge_gap_threshold: float = 0.3


class InsightFeedService:
    """
    Proactive intelligence service that generates insights from document corpus.

    Features:
    - Trending topic detection
    - Knowledge gap identification
    - Document connection suggestions
    - Personalized recommendations
    - Periodic digests
    """

    def __init__(
        self,
        config: Optional[InsightFeedConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        self.config = config or InsightFeedConfig()
        self.llm_client = llm_client
        self._user_profiles: Dict[str, UserActivityProfile] = {}
        self._insight_cache: Dict[str, List[Insight]] = {}
        self._last_analysis: Optional[datetime] = None

    async def generate_feed(
        self,
        user_id: str,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        limit: int = 20,
    ) -> List[Insight]:
        """
        Generate personalized insight feed for a user.

        Args:
            user_id: User to generate insights for
            documents: All documents in the corpus
            embeddings: Document embeddings (optional)
            limit: Maximum number of insights to return

        Returns:
            List of insights sorted by relevance
        """
        insights: List[Insight] = []

        # Get user profile
        profile = self._get_or_create_profile(user_id)

        # Generate different types of insights
        tasks = [
            self._detect_trending_topics(documents, embeddings),
            self._identify_knowledge_gaps(documents, embeddings),
            self._find_document_connections(documents, embeddings, profile),
            self._detect_stale_content(documents),
            self._detect_duplicates(documents, embeddings),
            self._generate_recommendations(documents, embeddings, profile),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning("Insight generation failed", error=str(result))
                continue
            insights.extend(result)

        # Personalize and rank insights
        insights = await self._rank_insights(insights, profile)

        # Filter dismissed insights
        insights = [i for i in insights if not i.is_dismissed]

        # Apply limit
        return insights[:limit]

    async def _detect_trending_topics(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray],
    ) -> List[Insight]:
        """Detect topics that are gaining attention recently."""
        insights = []
        cutoff = datetime.utcnow() - timedelta(days=self.config.trending_window_days)

        # Group documents by time
        recent_docs = []
        older_docs = []

        for doc in documents:
            created_at = self._parse_datetime(doc.get("created_at"))
            if created_at and created_at > cutoff:
                recent_docs.append(doc)
            else:
                older_docs.append(doc)

        if len(recent_docs) < self.config.min_documents_for_trend:
            return insights

        # Extract topics from recent documents
        recent_topics = self._extract_topics(recent_docs)
        older_topics = self._extract_topics(older_docs)

        # Find trending topics (more frequent recently)
        for topic, recent_count in recent_topics.items():
            older_count = older_topics.get(topic, 0)
            if older_count == 0 and recent_count >= self.config.min_documents_for_trend:
                # New trending topic
                insight = Insight(
                    id=self._generate_id("trend", topic),
                    insight_type=InsightType.TRENDING_TOPIC,
                    title=f"Emerging Topic: {topic.title()}",
                    description=f"'{topic}' has appeared in {recent_count} documents in the past {self.config.trending_window_days} days.",
                    priority=InsightPriority.MEDIUM,
                    relevance_score=min(1.0, recent_count / 10),
                    metadata={"topic": topic, "document_count": recent_count},
                    related_document_ids=[
                        d["id"] for d in recent_docs
                        if topic.lower() in (d.get("content", "") + d.get("title", "")).lower()
                    ][:5],
                    expires_at=datetime.utcnow() + timedelta(days=7),
                )
                insights.append(insight)

            elif older_count > 0:
                growth_rate = recent_count / max(older_count, 1)
                if growth_rate > 2:  # 2x growth
                    insight = Insight(
                        id=self._generate_id("trend", topic),
                        insight_type=InsightType.TRENDING_TOPIC,
                        title=f"Growing Interest: {topic.title()}",
                        description=f"'{topic}' mentions increased {growth_rate:.1f}x compared to before.",
                        priority=InsightPriority.MEDIUM,
                        relevance_score=min(1.0, growth_rate / 5),
                        metadata={
                            "topic": topic,
                            "growth_rate": growth_rate,
                            "recent_count": recent_count,
                            "older_count": older_count,
                        },
                        expires_at=datetime.utcnow() + timedelta(days=7),
                    )
                    insights.append(insight)

        return insights[:self.config.max_insights_per_type]

    async def _identify_knowledge_gaps(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray],
    ) -> List[Insight]:
        """Identify areas with sparse document coverage."""
        insights = []

        if embeddings is None or len(documents) < 10:
            return insights

        # Find sparse regions in embedding space
        # Calculate pairwise distances
        similarities = cosine_similarity(embeddings)

        # Documents with low average similarity to others might be isolated topics
        avg_similarities = np.mean(similarities, axis=1)
        threshold = np.percentile(avg_similarities, 25)  # Bottom 25%

        isolated_indices = np.where(avg_similarities < threshold)[0]

        for idx in isolated_indices[:self.config.max_insights_per_type]:
            doc = documents[idx]
            insight = Insight(
                id=self._generate_id("gap", doc["id"]),
                insight_type=InsightType.KNOWLEDGE_GAP,
                title=f"Isolated Topic: {doc.get('title', 'Untitled')[:50]}",
                description="This document covers a topic with limited related content. Consider adding more documentation.",
                priority=InsightPriority.LOW,
                relevance_score=1 - avg_similarities[idx],
                metadata={
                    "avg_similarity": float(avg_similarities[idx]),
                    "document_title": doc.get("title"),
                },
                related_document_ids=[doc["id"]],
                action_items=["Add related documents", "Link to existing content"],
            )
            insights.append(insight)

        return insights

    async def _find_document_connections(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray],
        profile: UserActivityProfile,
    ) -> List[Insight]:
        """Find connections between documents user has viewed."""
        insights = []

        if embeddings is None or not profile.viewed_documents:
            return insights

        # Get recently viewed document IDs
        recent_views = [
            doc_id for doc_id, viewed_at in profile.viewed_documents[-10:]
        ]

        doc_id_to_idx = {d["id"]: i for i, d in enumerate(documents)}

        for doc_id in recent_views:
            if doc_id not in doc_id_to_idx:
                continue

            idx = doc_id_to_idx[doc_id]
            doc = documents[idx]

            # Find similar documents
            similarities = cosine_similarity(
                embeddings[idx].reshape(1, -1),
                embeddings
            )[0]

            # Get top similar documents (excluding self)
            similar_indices = np.argsort(similarities)[::-1][1:6]

            for sim_idx in similar_indices:
                sim_score = similarities[sim_idx]
                if sim_score < self.config.similarity_threshold:
                    continue

                sim_doc = documents[sim_idx]

                # Don't suggest already viewed documents
                if sim_doc["id"] in recent_views:
                    continue

                insight = Insight(
                    id=self._generate_id("conn", f"{doc_id}_{sim_doc['id']}"),
                    insight_type=InsightType.DOCUMENT_CONNECTION,
                    title=f"Related: {sim_doc.get('title', 'Untitled')[:40]}",
                    description=f"Based on your interest in '{doc.get('title', 'Untitled')[:30]}', you might find this relevant.",
                    priority=InsightPriority.MEDIUM,
                    relevance_score=float(sim_score),
                    metadata={
                        "source_doc": doc.get("title"),
                        "similarity": float(sim_score),
                    },
                    related_document_ids=[doc_id, sim_doc["id"]],
                )
                insights.append(insight)

        return insights[:self.config.max_insights_per_type]

    async def _detect_stale_content(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Insight]:
        """Detect documents that may be outdated."""
        insights = []
        cutoff = datetime.utcnow() - timedelta(days=self.config.stale_threshold_days)

        stale_docs = []
        for doc in documents:
            updated_at = self._parse_datetime(
                doc.get("updated_at") or doc.get("created_at")
            )
            if updated_at and updated_at < cutoff:
                stale_docs.append((doc, updated_at))

        # Sort by oldest first
        stale_docs.sort(key=lambda x: x[1])

        for doc, updated_at in stale_docs[:self.config.max_insights_per_type]:
            days_old = (datetime.utcnow() - updated_at).days
            insight = Insight(
                id=self._generate_id("stale", doc["id"]),
                insight_type=InsightType.STALE_CONTENT,
                title=f"Review: {doc.get('title', 'Untitled')[:40]}",
                description=f"This document hasn't been updated in {days_old} days. Consider reviewing for accuracy.",
                priority=InsightPriority.LOW,
                relevance_score=min(1.0, days_old / 365),
                metadata={
                    "days_since_update": days_old,
                    "last_updated": updated_at.isoformat(),
                },
                related_document_ids=[doc["id"]],
                action_items=["Review content", "Update if needed", "Archive if obsolete"],
            )
            insights.append(insight)

        return insights

    async def _detect_duplicates(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray],
    ) -> List[Insight]:
        """Detect potentially duplicate documents."""
        insights = []

        if embeddings is None or len(documents) < 2:
            return insights

        # Find very similar documents
        similarities = cosine_similarity(embeddings)

        # Get upper triangle indices (avoid duplicate pairs)
        seen_pairs: Set[Tuple[str, str]] = set()

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                if similarities[i, j] > 0.95:  # Very high similarity
                    doc1, doc2 = documents[i], documents[j]
                    pair_key = tuple(sorted([doc1["id"], doc2["id"]]))

                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    insight = Insight(
                        id=self._generate_id("dup", f"{doc1['id']}_{doc2['id']}"),
                        insight_type=InsightType.DUPLICATE_DETECTION,
                        title="Potential Duplicate Documents",
                        description=f"'{doc1.get('title', 'Untitled')[:25]}' and '{doc2.get('title', 'Untitled')[:25]}' are very similar.",
                        priority=InsightPriority.MEDIUM,
                        relevance_score=float(similarities[i, j]),
                        metadata={
                            "similarity": float(similarities[i, j]),
                            "doc1_title": doc1.get("title"),
                            "doc2_title": doc2.get("title"),
                        },
                        related_document_ids=[doc1["id"], doc2["id"]],
                        action_items=["Review both documents", "Merge if duplicates", "Keep both if different"],
                    )
                    insights.append(insight)

        return insights[:self.config.max_insights_per_type]

    async def _generate_recommendations(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray],
        profile: UserActivityProfile,
    ) -> List[Insight]:
        """Generate personalized reading recommendations."""
        insights = []

        if not profile.frequent_topics:
            return insights

        # Find documents matching user's interests
        top_topics = sorted(
            profile.frequent_topics.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        viewed_ids = {doc_id for doc_id, _ in profile.viewed_documents}

        for topic, _ in top_topics:
            matching_docs = [
                d for d in documents
                if topic.lower() in (d.get("content", "") + d.get("title", "")).lower()
                and d["id"] not in viewed_ids
            ]

            if matching_docs:
                doc = matching_docs[0]  # Recommend first unread matching doc
                insight = Insight(
                    id=self._generate_id("rec", f"{topic}_{doc['id']}"),
                    insight_type=InsightType.READING_RECOMMENDATION,
                    title=f"Recommended: {doc.get('title', 'Untitled')[:40]}",
                    description=f"Based on your interest in '{topic}', this document might be helpful.",
                    priority=InsightPriority.LOW,
                    relevance_score=0.7,
                    metadata={"topic": topic, "match_reason": "topic_interest"},
                    related_document_ids=[doc["id"]],
                )
                insights.append(insight)

        return insights[:self.config.max_insights_per_type]

    async def generate_digest(
        self,
        user_id: str,
        documents: List[Dict[str, Any]],
        period: str = "daily",
    ) -> Insight:
        """Generate a summary digest for a time period."""
        cutoff_hours = 24 if period == "daily" else 168  # weekly
        cutoff = datetime.utcnow() - timedelta(hours=cutoff_hours)

        # Count new documents
        new_docs = [
            d for d in documents
            if self._parse_datetime(d.get("created_at")) and
            self._parse_datetime(d.get("created_at")) > cutoff
        ]

        # Count updated documents
        updated_docs = [
            d for d in documents
            if self._parse_datetime(d.get("updated_at")) and
            self._parse_datetime(d.get("updated_at")) > cutoff and
            d not in new_docs
        ]

        # Extract top topics
        topics = self._extract_topics(new_docs + updated_docs)
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]

        description = f"""
**{period.title()} Summary**

- {len(new_docs)} new documents added
- {len(updated_docs)} documents updated
- Top topics: {', '.join(t[0] for t in top_topics) if top_topics else 'None'}
        """.strip()

        return Insight(
            id=self._generate_id("digest", f"{user_id}_{period}"),
            insight_type=InsightType.SUMMARY_DIGEST,
            title=f"Your {period.title()} Digest",
            description=description,
            priority=InsightPriority.MEDIUM,
            relevance_score=0.8,
            metadata={
                "period": period,
                "new_document_count": len(new_docs),
                "updated_document_count": len(updated_docs),
                "top_topics": [t[0] for t in top_topics],
            },
            related_document_ids=[d["id"] for d in new_docs[:5]],
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(hours=cutoff_hours),
        )

    async def _rank_insights(
        self,
        insights: List[Insight],
        profile: UserActivityProfile,
    ) -> List[Insight]:
        """Rank insights by relevance to user."""
        # Priority weights
        priority_weights = {
            InsightPriority.CRITICAL: 4.0,
            InsightPriority.HIGH: 3.0,
            InsightPriority.MEDIUM: 2.0,
            InsightPriority.LOW: 1.0,
        }

        # Type preferences (can be personalized)
        type_weights = {
            InsightType.DOCUMENT_CONNECTION: 1.2,
            InsightType.READING_RECOMMENDATION: 1.1,
            InsightType.TRENDING_TOPIC: 1.0,
            InsightType.DUPLICATE_DETECTION: 0.9,
            InsightType.STALE_CONTENT: 0.8,
            InsightType.KNOWLEDGE_GAP: 0.7,
        }

        for insight in insights:
            base_score = insight.relevance_score
            priority_boost = priority_weights.get(insight.priority, 1.0)
            type_boost = type_weights.get(insight.insight_type, 1.0)

            # Recency boost
            age_hours = (datetime.utcnow() - insight.created_at).total_seconds() / 3600
            recency_boost = 1.0 / (1.0 + age_hours / 24)  # Decay over days

            insight.relevance_score = base_score * priority_boost * type_boost * recency_boost

        # Sort by final score
        insights.sort(key=lambda x: x.relevance_score, reverse=True)
        return insights

    def track_activity(
        self,
        user_id: str,
        activity_type: str,
        activity_data: Dict[str, Any],
    ) -> None:
        """Track user activity for personalization."""
        profile = self._get_or_create_profile(user_id)
        now = datetime.utcnow()

        if activity_type == "view":
            doc_id = activity_data.get("document_id")
            if doc_id:
                profile.viewed_documents.append((doc_id, now))
                # Keep last 100 views
                profile.viewed_documents = profile.viewed_documents[-100:]

        elif activity_type == "search":
            query = activity_data.get("query")
            if query:
                profile.searched_queries.append((query, now))
                # Extract topics from query
                for word in query.lower().split():
                    if len(word) > 3:
                        profile.frequent_topics[word] = profile.frequent_topics.get(word, 0) + 1

        elif activity_type == "create":
            doc_id = activity_data.get("document_id")
            if doc_id:
                profile.created_documents.append((doc_id, now))

        profile.last_active = now
        profile.active_hours[now.hour] = profile.active_hours.get(now.hour, 0) + 1

    def _get_or_create_profile(self, user_id: str) -> UserActivityProfile:
        """Get or create user activity profile."""
        if user_id not in self._user_profiles:
            self._user_profiles[user_id] = UserActivityProfile(user_id=user_id)
        return self._user_profiles[user_id]

    def _extract_topics(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Extract topic keywords from documents."""
        from collections import Counter
        import re

        topics: Counter = Counter()
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "this", "that", "it", "its", "as", "if", "when", "where", "which",
        }

        for doc in documents:
            text = doc.get("title", "") + " " + doc.get("content", "")[:500]
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            for word in words:
                if word not in stop_words:
                    topics[word] += 1

        return dict(topics.most_common(50))

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    def _generate_id(self, prefix: str, content: str) -> str:
        """Generate a unique insight ID."""
        data = f"{prefix}_{content}_{datetime.utcnow().date()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]


# Singleton instance
_insight_feed_service: Optional[InsightFeedService] = None


def get_insight_feed_service() -> InsightFeedService:
    """Get or create the insight feed service singleton."""
    global _insight_feed_service
    if _insight_feed_service is None:
        _insight_feed_service = InsightFeedService()
    return _insight_feed_service
