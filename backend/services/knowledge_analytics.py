"""
AIDocumentIndexer - Knowledge Analytics Service
================================================

Lynkt-inspired features for tracking knowledge usage and detecting patterns.

Features:
1. Knowledge Reuse Tracking - Monitor document/chunk usage across queries
2. Context Signal Detection - Identify missing links and anomalies
3. Usage Pattern Recognition - Track how knowledge is accessed
4. Knowledge Health Metrics - Measure knowledge base effectiveness
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import json

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class DocumentUsage:
    """Tracks usage of a single document."""
    document_id: str
    document_name: str
    total_retrievals: int = 0
    unique_queries: int = 0
    last_retrieved: Optional[datetime] = None
    avg_relevance_score: float = 0.0
    topics_matched: List[str] = field(default_factory=list)


@dataclass
class KnowledgeReuseSummary:
    """Summary of knowledge reuse metrics."""
    period_start: datetime
    period_end: datetime
    total_queries: int
    unique_documents_used: int
    total_retrievals: int
    avg_reuse_rate: float  # How often documents are reused
    top_documents: List[DocumentUsage]
    underutilized_documents: List[str]
    knowledge_coverage: float  # % of docs used at least once


@dataclass
class ContextSignal:
    """A detected context signal (gap or anomaly)."""
    signal_type: str  # "gap", "anomaly", "pattern", "stale"
    severity: str  # "low", "medium", "high"
    description: str
    affected_documents: List[str]
    suggested_action: str
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "severity": self.severity,
            "description": self.description,
            "affected_documents": self.affected_documents,
            "suggested_action": self.suggested_action,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class UsagePattern:
    """A detected usage pattern."""
    pattern_type: str  # "temporal", "topical", "user", "sequence"
    description: str
    frequency: int
    entities_involved: List[str]
    time_range: Optional[Tuple[datetime, datetime]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "frequency": self.frequency,
            "entities_involved": self.entities_involved,
            "time_range": [
                self.time_range[0].isoformat(),
                self.time_range[1].isoformat()
            ] if self.time_range else None,
        }


class KnowledgeAnalyticsService:
    """
    Service for tracking and analyzing knowledge usage.

    Provides:
    1. Document usage tracking
    2. Knowledge reuse metrics
    3. Context signal detection
    4. Usage pattern recognition
    """

    def __init__(self):
        """Initialize the analytics service."""
        # In-memory tracking (should be persisted to DB in production)
        self._document_usage: Dict[str, DocumentUsage] = {}
        self._query_history: List[Dict[str, Any]] = []
        self._retrieval_log: List[Dict[str, Any]] = []
        self._signals: List[ContextSignal] = []
        self._patterns: List[UsagePattern] = []

    async def track_retrieval(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        Track a document retrieval event.

        Args:
            query: The query that triggered retrieval
            retrieved_documents: List of retrieved document info
            user_id: Optional user ID
            session_id: Optional session ID
        """
        now = datetime.utcnow()

        # Log the query
        self._query_history.append({
            "query": query,
            "timestamp": now,
            "user_id": user_id,
            "session_id": session_id,
            "document_count": len(retrieved_documents),
        })

        # Update document usage
        for doc in retrieved_documents:
            doc_id = doc.get("document_id", doc.get("id", "unknown"))
            doc_name = doc.get("document_name", doc.get("title", "Unknown"))
            score = doc.get("score", doc.get("relevance_score", 0.5))

            if doc_id not in self._document_usage:
                self._document_usage[doc_id] = DocumentUsage(
                    document_id=doc_id,
                    document_name=doc_name,
                )

            usage = self._document_usage[doc_id]
            usage.total_retrievals += 1
            usage.last_retrieved = now
            usage.avg_relevance_score = (
                (usage.avg_relevance_score * (usage.total_retrievals - 1) + score)
                / usage.total_retrievals
            )

            # Track unique queries (simplified)
            usage.unique_queries = min(usage.unique_queries + 1, usage.total_retrievals)

            # Log retrieval
            self._retrieval_log.append({
                "document_id": doc_id,
                "query": query,
                "score": score,
                "timestamp": now,
                "user_id": user_id,
            })

        # Trim old history
        cutoff = now - timedelta(days=30)
        self._query_history = [q for q in self._query_history if q["timestamp"] > cutoff]
        self._retrieval_log = [r for r in self._retrieval_log if r["timestamp"] > cutoff]

        logger.debug(
            "Tracked retrieval",
            query_length=len(query),
            document_count=len(retrieved_documents),
        )

    async def get_reuse_summary(
        self,
        period_days: int = 7,
        top_n: int = 10,
    ) -> KnowledgeReuseSummary:
        """
        Get a summary of knowledge reuse for a period.

        Args:
            period_days: Number of days to analyze
            top_n: Number of top documents to include

        Returns:
            KnowledgeReuseSummary with metrics
        """
        now = datetime.utcnow()
        period_start = now - timedelta(days=period_days)

        # Filter to period
        period_queries = [
            q for q in self._query_history
            if q["timestamp"] >= period_start
        ]

        period_retrievals = [
            r for r in self._retrieval_log
            if r["timestamp"] >= period_start
        ]

        # Calculate metrics
        total_queries = len(period_queries)
        docs_used = set(r["document_id"] for r in period_retrievals)
        total_retrievals = len(period_retrievals)

        # Calculate reuse rate
        retrieval_counts = defaultdict(int)
        for r in period_retrievals:
            retrieval_counts[r["document_id"]] += 1

        avg_reuse = (
            sum(retrieval_counts.values()) / len(retrieval_counts)
            if retrieval_counts else 0
        )

        # Top documents
        top_docs = sorted(
            self._document_usage.values(),
            key=lambda d: d.total_retrievals,
            reverse=True,
        )[:top_n]

        # Underutilized documents (retrieved but low relevance)
        underutilized = [
            d.document_name for d in self._document_usage.values()
            if d.avg_relevance_score < 0.3 and d.total_retrievals > 0
        ][:10]

        # Knowledge coverage (rough estimate)
        total_docs = len(self._document_usage) if self._document_usage else 1
        coverage = len(docs_used) / total_docs if total_docs > 0 else 0

        return KnowledgeReuseSummary(
            period_start=period_start,
            period_end=now,
            total_queries=total_queries,
            unique_documents_used=len(docs_used),
            total_retrievals=total_retrievals,
            avg_reuse_rate=avg_reuse,
            top_documents=top_docs,
            underutilized_documents=underutilized,
            knowledge_coverage=coverage,
        )

    async def detect_signals(
        self,
        all_document_ids: List[str] = None,
    ) -> List[ContextSignal]:
        """
        Detect context signals (gaps, anomalies, patterns).

        Args:
            all_document_ids: Optional list of all document IDs for gap detection

        Returns:
            List of detected signals
        """
        signals = []
        now = datetime.utcnow()

        # 1. Detect stale documents (not retrieved recently)
        stale_threshold = now - timedelta(days=30)
        stale_docs = [
            d for d in self._document_usage.values()
            if d.last_retrieved and d.last_retrieved < stale_threshold
            and d.total_retrievals > 5  # Was popular but now stale
        ]

        if stale_docs:
            signals.append(ContextSignal(
                signal_type="stale",
                severity="medium",
                description=f"{len(stale_docs)} previously popular documents haven't been retrieved in 30+ days",
                affected_documents=[d.document_name for d in stale_docs[:10]],
                suggested_action="Review if these documents are still relevant or need updating",
            ))

        # 2. Detect low-relevance patterns
        low_relevance = [
            d for d in self._document_usage.values()
            if d.avg_relevance_score < 0.4 and d.total_retrievals > 10
        ]

        if low_relevance:
            signals.append(ContextSignal(
                signal_type="anomaly",
                severity="low",
                description=f"{len(low_relevance)} documents are frequently retrieved but with low relevance scores",
                affected_documents=[d.document_name for d in low_relevance[:10]],
                suggested_action="Consider improving document chunking or metadata for better matching",
            ))

        # 3. Detect knowledge gaps (documents never used)
        if all_document_ids:
            used_ids = set(self._document_usage.keys())
            unused_ids = set(all_document_ids) - used_ids

            if len(unused_ids) > len(all_document_ids) * 0.3:
                signals.append(ContextSignal(
                    signal_type="gap",
                    severity="high",
                    description=f"{len(unused_ids)} documents ({int(len(unused_ids)/len(all_document_ids)*100)}%) have never been retrieved",
                    affected_documents=list(unused_ids)[:20],
                    suggested_action="Review unused documents - they may need better indexing or may not be relevant",
                ))

        # 4. Detect query patterns without matches
        failed_queries = [
            q for q in self._query_history
            if q.get("document_count", 0) == 0
        ]

        if len(failed_queries) > len(self._query_history) * 0.1:
            signals.append(ContextSignal(
                signal_type="gap",
                severity="medium",
                description=f"{len(failed_queries)} queries returned no documents",
                affected_documents=[],
                suggested_action="Analyze failed queries to identify missing knowledge areas",
            ))

        self._signals = signals
        return signals

    async def detect_patterns(self) -> List[UsagePattern]:
        """
        Detect usage patterns in the knowledge base.

        Returns:
            List of detected patterns
        """
        patterns = []
        now = datetime.utcnow()

        # 1. Temporal patterns (peak usage times)
        if self._query_history:
            hour_counts = defaultdict(int)
            for q in self._query_history:
                hour = q["timestamp"].hour
                hour_counts[hour] += 1

            peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 0
            peak_count = hour_counts.get(peak_hour, 0)

            if peak_count > len(self._query_history) * 0.2:
                patterns.append(UsagePattern(
                    pattern_type="temporal",
                    description=f"Peak usage at {peak_hour}:00 ({peak_count} queries)",
                    frequency=peak_count,
                    entities_involved=[f"hour_{peak_hour}"],
                ))

        # 2. Document co-occurrence patterns
        if self._retrieval_log:
            # Group retrievals by query
            query_docs = defaultdict(set)
            for r in self._retrieval_log:
                query_docs[r["query"][:50]].add(r["document_id"])

            # Find frequently co-occurring documents
            co_occurrence = defaultdict(int)
            for docs in query_docs.values():
                doc_list = list(docs)
                for i in range(len(doc_list)):
                    for j in range(i + 1, len(doc_list)):
                        pair = tuple(sorted([doc_list[i], doc_list[j]]))
                        co_occurrence[pair] += 1

            # Top co-occurring pairs
            top_pairs = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:5]
            for pair, count in top_pairs:
                if count >= 5:
                    patterns.append(UsagePattern(
                        pattern_type="topical",
                        description=f"Documents frequently retrieved together ({count} times)",
                        frequency=count,
                        entities_involved=list(pair),
                    ))

        # 3. User patterns
        if self._query_history:
            user_counts = defaultdict(int)
            for q in self._query_history:
                if q.get("user_id"):
                    user_counts[q["user_id"]] += 1

            for user_id, count in user_counts.items():
                if count > len(self._query_history) * 0.3:
                    patterns.append(UsagePattern(
                        pattern_type="user",
                        description=f"High activity user with {count} queries",
                        frequency=count,
                        entities_involved=[user_id],
                    ))

        self._patterns = patterns
        return patterns

    async def get_knowledge_health(self) -> Dict[str, Any]:
        """
        Get overall knowledge base health metrics.

        Returns:
            Dictionary of health metrics
        """
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)

        # Calculate metrics
        recent_queries = [q for q in self._query_history if q["timestamp"] >= week_ago]
        recent_retrievals = [r for r in self._retrieval_log if r["timestamp"] >= week_ago]

        total_docs = len(self._document_usage)
        active_docs = len([
            d for d in self._document_usage.values()
            if d.last_retrieved and d.last_retrieved >= week_ago
        ])

        avg_relevance = (
            sum(d.avg_relevance_score for d in self._document_usage.values())
            / total_docs if total_docs > 0 else 0
        )

        reuse_count = sum(d.total_retrievals for d in self._document_usage.values())

        return {
            "total_documents": total_docs,
            "active_documents_7d": active_docs,
            "activity_rate": active_docs / total_docs if total_docs > 0 else 0,
            "queries_7d": len(recent_queries),
            "retrievals_7d": len(recent_retrievals),
            "avg_relevance_score": round(avg_relevance, 3),
            "total_reuse_count": reuse_count,
            "avg_reuse_per_doc": round(reuse_count / total_docs, 2) if total_docs > 0 else 0,
            "signals_detected": len(self._signals),
            "patterns_detected": len(self._patterns),
            "health_score": self._calculate_health_score(
                active_docs / total_docs if total_docs > 0 else 0,
                avg_relevance,
                len(self._signals),
            ),
        }

    def _calculate_health_score(
        self,
        activity_rate: float,
        avg_relevance: float,
        signal_count: int,
    ) -> float:
        """Calculate overall health score (0-100)."""
        # Base score from activity and relevance
        base_score = (activity_rate * 40) + (avg_relevance * 40)

        # Deduct for signals
        signal_penalty = min(signal_count * 5, 20)

        return max(0, min(100, base_score + 20 - signal_penalty))

    def get_document_usage(self, document_id: str) -> Optional[DocumentUsage]:
        """Get usage stats for a specific document."""
        return self._document_usage.get(document_id)

    async def get_usage_report(self) -> Dict[str, Any]:
        """Generate a comprehensive usage report."""
        summary = await self.get_reuse_summary()
        signals = await self.detect_signals()
        patterns = await self.detect_patterns()
        health = await self.get_knowledge_health()

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start": summary.period_start.isoformat(),
                "end": summary.period_end.isoformat(),
            },
            "health": health,
            "usage_summary": {
                "total_queries": summary.total_queries,
                "unique_documents_used": summary.unique_documents_used,
                "total_retrievals": summary.total_retrievals,
                "avg_reuse_rate": summary.avg_reuse_rate,
                "knowledge_coverage": summary.knowledge_coverage,
            },
            "top_documents": [
                {
                    "name": d.document_name,
                    "retrievals": d.total_retrievals,
                    "avg_relevance": d.avg_relevance_score,
                }
                for d in summary.top_documents[:5]
            ],
            "signals": [s.to_dict() for s in signals],
            "patterns": [p.to_dict() for p in patterns],
        }


# Singleton instance
_knowledge_analytics: Optional[KnowledgeAnalyticsService] = None


def get_knowledge_analytics_service() -> KnowledgeAnalyticsService:
    """Get or create the knowledge analytics service singleton."""
    global _knowledge_analytics
    if _knowledge_analytics is None:
        _knowledge_analytics = KnowledgeAnalyticsService()
    return _knowledge_analytics
