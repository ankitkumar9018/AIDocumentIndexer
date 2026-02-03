"""
AIDocumentIndexer - Conflict Detector Service
===============================================

Consistency checker that:
- Detects contradictory information across documents
- Identifies outdated information
- Finds conflicting facts and statements
- Tracks information provenance
- Suggests resolutions
"""

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib

import structlog
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger(__name__)


class ConflictType(str, Enum):
    """Types of detected conflicts."""
    FACTUAL = "factual"  # Contradictory facts
    NUMERICAL = "numerical"  # Different numbers for same metric
    TEMPORAL = "temporal"  # Outdated vs current information
    DEFINITIONAL = "definitional"  # Different definitions
    PROCEDURAL = "procedural"  # Conflicting procedures
    ATTRIBUTION = "attribution"  # Different sources for same claim
    VERSION = "version"  # Version inconsistencies


class ConflictSeverity(str, Enum):
    """Severity levels for conflicts."""
    CRITICAL = "critical"  # Must be resolved immediately
    HIGH = "high"  # Should be resolved soon
    MEDIUM = "medium"  # Should be reviewed
    LOW = "low"  # Minor inconsistency
    INFO = "info"  # Informational, may not need action


class ResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts."""
    USE_NEWER = "use_newer"  # Use more recent information
    USE_AUTHORITATIVE = "use_authoritative"  # Use authoritative source
    MERGE = "merge"  # Combine information
    MANUAL_REVIEW = "manual_review"  # Requires human decision
    FLAG_BOTH = "flag_both"  # Keep both with warning
    DEPRECATE_OLDER = "deprecate_older"  # Mark older as deprecated


@dataclass
class ConflictingStatement:
    """A statement involved in a conflict."""
    text: str
    document_id: str
    document_title: str
    source_date: Optional[datetime] = None
    confidence: float = 1.0
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "source_date": self.source_date.isoformat() if self.source_date else None,
            "confidence": self.confidence,
            "context": self.context,
            "metadata": self.metadata,
        }


@dataclass
class Conflict:
    """Represents a detected conflict."""
    id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    statements: List[ConflictingStatement]
    suggested_resolution: ResolutionStrategy
    resolution_explanation: str
    topic: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution_notes: str = ""
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "statements": [s.to_dict() for s in self.statements],
            "suggested_resolution": self.suggested_resolution.value,
            "resolution_explanation": self.resolution_explanation,
            "topic": self.topic,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "document_count": len(set(s.document_id for s in self.statements)),
        }


@dataclass
class ConflictReport:
    """Summary report of all detected conflicts."""
    total_conflicts: int
    conflicts_by_type: Dict[str, int]
    conflicts_by_severity: Dict[str, int]
    conflicts: List[Conflict]
    analyzed_documents: int
    analysis_time_ms: float
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_conflicts": self.total_conflicts,
            "conflicts_by_type": self.conflicts_by_type,
            "conflicts_by_severity": self.conflicts_by_severity,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "analyzed_documents": self.analyzed_documents,
            "analysis_time_ms": self.analysis_time_ms,
            "recommendations": self.recommendations,
        }


class ConflictDetectorService:
    """
    Service for detecting conflicting information across documents.

    Features:
    - Factual contradiction detection
    - Numerical inconsistency detection
    - Temporal conflict identification
    - LLM-powered semantic analysis
    - Resolution suggestions
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        similarity_threshold: float = 0.8,
        numerical_tolerance: float = 0.05,
    ):
        self.llm_client = llm_client
        self.similarity_threshold = similarity_threshold
        self.numerical_tolerance = numerical_tolerance
        self._conflict_cache: Dict[str, List[Conflict]] = {}

    async def analyze_corpus(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        focus_topics: Optional[List[str]] = None,
    ) -> ConflictReport:
        """
        Analyze entire corpus for conflicts.

        Args:
            documents: List of documents with id, title, content, metadata
            embeddings: Document embeddings for similarity
            focus_topics: Optional topics to focus on

        Returns:
            ConflictReport with all detected conflicts
        """
        start_time = datetime.utcnow()
        conflicts: List[Conflict] = []

        # Extract statements from documents
        statements = await self._extract_statements(documents)

        # Group statements by topic for comparison
        topic_groups = self._group_by_topic(statements)

        # Detect different types of conflicts
        tasks = [
            self._detect_factual_conflicts(topic_groups, embeddings),
            self._detect_numerical_conflicts(statements),
            self._detect_temporal_conflicts(statements),
            self._detect_definitional_conflicts(topic_groups),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning("Conflict detection failed", error=str(result))
                continue
            conflicts.extend(result)

        # Deduplicate conflicts
        conflicts = self._deduplicate_conflicts(conflicts)

        # Sort by severity
        severity_order = {
            ConflictSeverity.CRITICAL: 0,
            ConflictSeverity.HIGH: 1,
            ConflictSeverity.MEDIUM: 2,
            ConflictSeverity.LOW: 3,
            ConflictSeverity.INFO: 4,
        }
        conflicts.sort(key=lambda c: severity_order.get(c.severity, 5))

        # Generate report
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        conflicts_by_type = defaultdict(int)
        conflicts_by_severity = defaultdict(int)
        for conflict in conflicts:
            conflicts_by_type[conflict.conflict_type.value] += 1
            conflicts_by_severity[conflict.severity.value] += 1

        recommendations = self._generate_recommendations(conflicts)

        return ConflictReport(
            total_conflicts=len(conflicts),
            conflicts_by_type=dict(conflicts_by_type),
            conflicts_by_severity=dict(conflicts_by_severity),
            conflicts=conflicts,
            analyzed_documents=len(documents),
            analysis_time_ms=processing_time,
            recommendations=recommendations,
        )

    async def check_document(
        self,
        new_document: Dict[str, Any],
        existing_documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
    ) -> List[Conflict]:
        """
        Check a new document against existing corpus for conflicts.

        Args:
            new_document: Document to check
            existing_documents: Existing corpus
            embeddings: Embeddings for similarity

        Returns:
            List of conflicts with existing documents
        """
        conflicts: List[Conflict] = []

        # Extract statements from new document
        new_statements = await self._extract_statements_from_document(new_document)

        # Extract statements from existing documents
        existing_statements = []
        for doc in existing_documents:
            existing_statements.extend(
                await self._extract_statements_from_document(doc)
            )

        # Compare new statements against existing
        for new_stmt in new_statements:
            for existing_stmt in existing_statements:
                conflict = await self._compare_statements(new_stmt, existing_stmt)
                if conflict:
                    conflicts.append(conflict)

        return self._deduplicate_conflicts(conflicts)

    async def _extract_statements(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract factual statements from documents."""
        all_statements = []

        for doc in documents:
            statements = await self._extract_statements_from_document(doc)
            all_statements.extend(statements)

        return all_statements

    async def _extract_statements_from_document(
        self,
        document: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract statements from a single document."""
        content = document.get("content", "")
        title = document.get("title", "Untitled")
        doc_id = document.get("id", "")
        doc_date = self._parse_date(document.get("created_at") or document.get("updated_at"))

        statements = []

        # Extract sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            # Classify statement type
            stmt_type = self._classify_statement(sentence)
            if not stmt_type:
                continue

            # Extract topic
            topic = self._extract_topic(sentence)

            statements.append({
                "text": sentence,
                "document_id": doc_id,
                "document_title": title,
                "source_date": doc_date,
                "type": stmt_type,
                "topic": topic,
                "metadata": document.get("metadata", {}),
            })

        return statements

    def _classify_statement(self, sentence: str) -> Optional[str]:
        """Classify statement type for conflict detection."""
        lower = sentence.lower()

        # Factual statements (definitions, facts)
        if re.search(r'\b(is|are|was|were|has|have|equals?)\b', lower):
            return "factual"

        # Numerical statements
        if re.search(r'\d+(?:\.\d+)?(?:%|percent|million|billion|thousand)?', lower):
            return "numerical"

        # Procedural statements
        if re.search(r'\b(should|must|always|never|first|then|before|after)\b', lower):
            return "procedural"

        # Temporal statements
        if re.search(r'\b(since|until|starting|ending|as of|current|latest)\b', lower):
            return "temporal"

        return None

    def _extract_topic(self, sentence: str) -> str:
        """Extract main topic from a sentence."""
        # Find capitalized phrases (potential topics)
        topics = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', sentence)
        if topics:
            return topics[0]

        # Fall back to first noun phrase
        words = sentence.split()[:5]
        return ' '.join(words)

    def _group_by_topic(
        self,
        statements: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group statements by topic for comparison."""
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for stmt in statements:
            topic = stmt.get("topic", "unknown")
            groups[topic.lower()].append(stmt)

        return groups

    async def _detect_factual_conflicts(
        self,
        topic_groups: Dict[str, List[Dict[str, Any]]],
        embeddings: Optional[np.ndarray],
    ) -> List[Conflict]:
        """Detect factual contradictions."""
        conflicts = []

        for topic, statements in topic_groups.items():
            if len(statements) < 2:
                continue

            # Compare all pairs of statements about same topic
            for i, stmt1 in enumerate(statements):
                for stmt2 in statements[i + 1:]:
                    # Skip if from same document
                    if stmt1["document_id"] == stmt2["document_id"]:
                        continue

                    conflict = await self._check_factual_conflict(stmt1, stmt2, topic)
                    if conflict:
                        conflicts.append(conflict)

        return conflicts

    async def _check_factual_conflict(
        self,
        stmt1: Dict[str, Any],
        stmt2: Dict[str, Any],
        topic: str,
    ) -> Optional[Conflict]:
        """Check if two statements are factually conflicting."""
        # Use LLM for semantic conflict detection
        if self.llm_client:
            try:
                prompt = f"""Determine if these two statements about "{topic}" contradict each other.

Statement 1: {stmt1['text']}
Statement 2: {stmt2['text']}

Respond in JSON format:
{{
  "contradicts": true/false,
  "explanation": "brief explanation",
  "severity": "critical/high/medium/low"
}}"""

                response = await self.llm_client.complete(prompt)
                import json
                result = json.loads(response)

                if result.get("contradicts"):
                    severity_map = {
                        "critical": ConflictSeverity.CRITICAL,
                        "high": ConflictSeverity.HIGH,
                        "medium": ConflictSeverity.MEDIUM,
                        "low": ConflictSeverity.LOW,
                    }
                    severity = severity_map.get(
                        result.get("severity", "medium"),
                        ConflictSeverity.MEDIUM
                    )

                    return self._create_conflict(
                        conflict_type=ConflictType.FACTUAL,
                        severity=severity,
                        description=result.get("explanation", "Contradictory statements detected"),
                        statements=[stmt1, stmt2],
                        topic=topic,
                    )

            except Exception as e:
                logger.warning("LLM conflict check failed", error=str(e))

        # Fallback: heuristic check
        return self._heuristic_conflict_check(stmt1, stmt2, topic)

    def _heuristic_conflict_check(
        self,
        stmt1: Dict[str, Any],
        stmt2: Dict[str, Any],
        topic: str,
    ) -> Optional[Conflict]:
        """Heuristic-based conflict detection."""
        text1 = stmt1["text"].lower()
        text2 = stmt2["text"].lower()

        # Check for negation patterns
        negation_pairs = [
            ("is not", "is"),
            ("does not", "does"),
            ("cannot", "can"),
            ("never", "always"),
            ("false", "true"),
            ("incorrect", "correct"),
        ]

        for neg, pos in negation_pairs:
            if (neg in text1 and pos in text2) or (pos in text1 and neg in text2):
                # Check if they're about the same subject
                words1 = set(text1.split())
                words2 = set(text2.split())
                overlap = len(words1 & words2) / min(len(words1), len(words2))

                if overlap > 0.3:
                    return self._create_conflict(
                        conflict_type=ConflictType.FACTUAL,
                        severity=ConflictSeverity.MEDIUM,
                        description=f"Potential contradiction detected about '{topic}'",
                        statements=[stmt1, stmt2],
                        topic=topic,
                    )

        return None

    async def _detect_numerical_conflicts(
        self,
        statements: List[Dict[str, Any]],
    ) -> List[Conflict]:
        """Detect numerical inconsistencies."""
        conflicts = []

        # Group statements with numbers by topic
        numerical_statements: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for stmt in statements:
            if stmt.get("type") == "numerical":
                numbers = re.findall(r'\d+(?:\.\d+)?', stmt["text"])
                if numbers:
                    stmt["numbers"] = [float(n) for n in numbers]
                    topic = stmt.get("topic", "").lower()
                    numerical_statements[topic].append(stmt)

        # Compare numbers about same topic
        for topic, stmts in numerical_statements.items():
            if len(stmts) < 2:
                continue

            for i, stmt1 in enumerate(stmts):
                for stmt2 in stmts[i + 1:]:
                    if stmt1["document_id"] == stmt2["document_id"]:
                        continue

                    # Compare numbers
                    for n1 in stmt1.get("numbers", []):
                        for n2 in stmt2.get("numbers", []):
                            if n1 > 0 and n2 > 0:
                                diff = abs(n1 - n2) / max(n1, n2)
                                if diff > self.numerical_tolerance:
                                    conflicts.append(self._create_conflict(
                                        conflict_type=ConflictType.NUMERICAL,
                                        severity=self._assess_numerical_severity(diff),
                                        description=f"Different values for '{topic}': {n1} vs {n2}",
                                        statements=[stmt1, stmt2],
                                        topic=topic,
                                    ))

        return conflicts

    def _assess_numerical_severity(self, difference: float) -> ConflictSeverity:
        """Assess severity based on numerical difference."""
        if difference > 0.5:  # >50% difference
            return ConflictSeverity.CRITICAL
        elif difference > 0.2:  # >20% difference
            return ConflictSeverity.HIGH
        elif difference > 0.1:  # >10% difference
            return ConflictSeverity.MEDIUM
        else:
            return ConflictSeverity.LOW

    async def _detect_temporal_conflicts(
        self,
        statements: List[Dict[str, Any]],
    ) -> List[Conflict]:
        """Detect temporal inconsistencies (outdated information)."""
        conflicts = []

        # Group by topic and sort by date
        topic_dated: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for stmt in statements:
            if stmt.get("source_date"):
                topic = stmt.get("topic", "").lower()
                topic_dated[topic].append(stmt)

        for topic, stmts in topic_dated.items():
            if len(stmts) < 2:
                continue

            # Sort by date
            stmts.sort(key=lambda s: s["source_date"], reverse=True)

            # Compare newer vs older statements
            newest = stmts[0]
            for older in stmts[1:]:
                if oldest["document_id"] == newest["document_id"]:
                    continue

                # Check if content differs significantly
                if self._texts_differ_significantly(newest["text"], older["text"]):
                    days_diff = (newest["source_date"] - older["source_date"]).days

                    if days_diff > 30:  # More than 30 days old
                        conflicts.append(self._create_conflict(
                            conflict_type=ConflictType.TEMPORAL,
                            severity=ConflictSeverity.MEDIUM,
                            description=f"Potentially outdated information about '{topic}' ({days_diff} days difference)",
                            statements=[newest, older],
                            topic=topic,
                        ))

        return conflicts

    async def _detect_definitional_conflicts(
        self,
        topic_groups: Dict[str, List[Dict[str, Any]]],
    ) -> List[Conflict]:
        """Detect conflicting definitions."""
        conflicts = []

        # Look for definition patterns
        definition_pattern = r'(?:is defined as|means|refers to|is)\s+(.+)'

        for topic, statements in topic_groups.items():
            definitions = []

            for stmt in statements:
                match = re.search(definition_pattern, stmt["text"], re.IGNORECASE)
                if match:
                    definitions.append({
                        **stmt,
                        "definition": match.group(1),
                    })

            # Compare definitions
            if len(definitions) >= 2:
                for i, def1 in enumerate(definitions):
                    for def2 in definitions[i + 1:]:
                        if def1["document_id"] == def2["document_id"]:
                            continue

                        if self._texts_differ_significantly(
                            def1["definition"],
                            def2["definition"]
                        ):
                            conflicts.append(self._create_conflict(
                                conflict_type=ConflictType.DEFINITIONAL,
                                severity=ConflictSeverity.HIGH,
                                description=f"Different definitions for '{topic}'",
                                statements=[def1, def2],
                                topic=topic,
                            ))

        return conflicts

    def _texts_differ_significantly(self, text1: str, text2: str) -> bool:
        """Check if two texts differ significantly."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return True

        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return overlap < 0.5

    async def _compare_statements(
        self,
        stmt1: Dict[str, Any],
        stmt2: Dict[str, Any],
    ) -> Optional[Conflict]:
        """Compare two statements for conflict."""
        # Same topic check
        if stmt1.get("topic", "").lower() != stmt2.get("topic", "").lower():
            return None

        topic = stmt1.get("topic", "unknown")
        return await self._check_factual_conflict(stmt1, stmt2, topic)

    def _create_conflict(
        self,
        conflict_type: ConflictType,
        severity: ConflictSeverity,
        description: str,
        statements: List[Dict[str, Any]],
        topic: str,
    ) -> Conflict:
        """Create a conflict instance."""
        # Determine resolution strategy
        resolution, explanation = self._suggest_resolution(
            conflict_type, statements
        )

        conflict_statements = [
            ConflictingStatement(
                text=s["text"],
                document_id=s["document_id"],
                document_title=s["document_title"],
                source_date=s.get("source_date"),
                confidence=1.0,
                metadata=s.get("metadata", {}),
            )
            for s in statements
        ]

        return Conflict(
            id=self._generate_id(conflict_type, statements),
            conflict_type=conflict_type,
            severity=severity,
            description=description,
            statements=conflict_statements,
            suggested_resolution=resolution,
            resolution_explanation=explanation,
            topic=topic,
        )

    def _suggest_resolution(
        self,
        conflict_type: ConflictType,
        statements: List[Dict[str, Any]],
    ) -> Tuple[ResolutionStrategy, str]:
        """Suggest resolution strategy."""
        # Sort by date if available
        dated_stmts = [s for s in statements if s.get("source_date")]

        if dated_stmts:
            dated_stmts.sort(key=lambda s: s["source_date"], reverse=True)

        if conflict_type == ConflictType.TEMPORAL:
            return (
                ResolutionStrategy.USE_NEWER,
                "Use the more recent information and mark older as outdated."
            )

        elif conflict_type == ConflictType.NUMERICAL:
            if len(dated_stmts) >= 2:
                return (
                    ResolutionStrategy.USE_NEWER,
                    "Use the most recent numerical value."
                )
            return (
                ResolutionStrategy.MANUAL_REVIEW,
                "Verify the correct value from authoritative source."
            )

        elif conflict_type == ConflictType.DEFINITIONAL:
            return (
                ResolutionStrategy.MANUAL_REVIEW,
                "Review both definitions and create a canonical definition."
            )

        elif conflict_type == ConflictType.FACTUAL:
            return (
                ResolutionStrategy.MANUAL_REVIEW,
                "Investigate which statement is accurate and update accordingly."
            )

        return (
            ResolutionStrategy.FLAG_BOTH,
            "Both statements flagged for review."
        )

    def _deduplicate_conflicts(self, conflicts: List[Conflict]) -> List[Conflict]:
        """Remove duplicate conflicts."""
        seen: Set[str] = set()
        unique = []

        for conflict in conflicts:
            # Create signature from document IDs and topic
            doc_ids = tuple(sorted(s.document_id for s in conflict.statements))
            signature = f"{conflict.topic}_{doc_ids}"

            if signature not in seen:
                seen.add(signature)
                unique.append(conflict)

        return unique

    def _generate_recommendations(self, conflicts: List[Conflict]) -> List[str]:
        """Generate recommendations based on conflicts."""
        recommendations = []

        if not conflicts:
            recommendations.append("No conflicts detected. Documentation is consistent.")
            return recommendations

        # Count by type
        type_counts = defaultdict(int)
        for c in conflicts:
            type_counts[c.conflict_type] += 1

        if type_counts[ConflictType.TEMPORAL] > 3:
            recommendations.append(
                f"Found {type_counts[ConflictType.TEMPORAL]} temporal conflicts. "
                "Consider implementing a document review schedule."
            )

        if type_counts[ConflictType.NUMERICAL] > 2:
            recommendations.append(
                f"Found {type_counts[ConflictType.NUMERICAL]} numerical inconsistencies. "
                "Consider establishing a single source of truth for metrics."
            )

        if type_counts[ConflictType.DEFINITIONAL] > 1:
            recommendations.append(
                f"Found {type_counts[ConflictType.DEFINITIONAL]} definition conflicts. "
                "Consider creating a glossary with canonical definitions."
            )

        # Severity-based recommendations
        critical = [c for c in conflicts if c.severity == ConflictSeverity.CRITICAL]
        if critical:
            recommendations.append(
                f"URGENT: {len(critical)} critical conflicts require immediate attention."
            )

        return recommendations

    def _generate_id(
        self,
        conflict_type: ConflictType,
        statements: List[Dict[str, Any]],
    ) -> str:
        """Generate unique conflict ID."""
        content = f"{conflict_type.value}_{'_'.join(s['document_id'] for s in statements)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _parse_date(self, value: Any) -> Optional[datetime]:
        """Parse date from various formats."""
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

    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution: ResolutionStrategy,
        notes: str,
        resolved_by: str,
    ) -> bool:
        """Mark a conflict as resolved."""
        # In production, update database
        logger.info(
            "Conflict resolved",
            conflict_id=conflict_id,
            resolution=resolution.value,
            resolved_by=resolved_by,
        )
        return True


# Singleton instance
_conflict_detector_service: Optional[ConflictDetectorService] = None


def get_conflict_detector_service() -> ConflictDetectorService:
    """Get or create the conflict detector service singleton."""
    global _conflict_detector_service
    if _conflict_detector_service is None:
        _conflict_detector_service = ConflictDetectorService()
    return _conflict_detector_service
