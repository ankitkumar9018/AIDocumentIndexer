"""
AIDocumentIndexer - Learning-to-Rank Service (Phase 65)
=======================================================

XGBoost-based Learning-to-Rank for search-engine quality ranking.

LTR trains a model on user feedback (clicks, dwell time, explicit ratings)
to learn optimal ranking that accounts for multiple features:
- BM25 score
- Semantic similarity
- Title/field matches
- Freshness
- Document quality signals

Research:
- LambdaMART: Pairwise ranking (Microsoft Learning to Rank)
- XGBoost Ranking: rank:pairwise, rank:ndcg objectives
- Click Models: Position bias correction

Usage:
    from backend.services.learning_to_rank import (
        LTRRanker,
        get_ltr_ranker,
    )

    ranker = get_ltr_ranker()

    # Record feedback
    await ranker.record_feedback(query, doc_id, clicked=True, dwell_time=45)

    # Train model
    await ranker.train()

    # Rerank results
    results = await ranker.rerank(query, candidates)
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog

logger = structlog.get_logger(__name__)

# NumPy for features
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# XGBoost for ranking
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

# Scikit-learn for preprocessing
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    StandardScaler = None


# =============================================================================
# Configuration
# =============================================================================

class LTRObjective(str, Enum):
    """XGBoost ranking objectives."""
    PAIRWISE = "rank:pairwise"  # LambdaRank
    NDCG = "rank:ndcg"  # NDCG optimization
    MAP = "rank:map"  # Mean Average Precision


@dataclass
class LTRConfig:
    """Configuration for Learning-to-Rank."""
    # Model settings
    objective: LTRObjective = LTRObjective.PAIRWISE
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1

    # Feature settings
    feature_names: List[str] = field(default_factory=lambda: [
        "bm25_score",
        "semantic_similarity",
        "title_match",
        "section_match",
        "freshness",
        "chunk_position",
        "term_proximity",
        "doc_length",
    ])

    # Training settings
    min_training_samples: int = 100  # Minimum samples to train
    validation_split: float = 0.2
    early_stopping_rounds: int = 10

    # Feedback settings
    click_weight: float = 1.0  # Weight for click signals
    dwell_weight: float = 2.0  # Weight for dwell time signals
    explicit_weight: float = 3.0  # Weight for explicit ratings

    # Persistence
    model_dir: str = "data/ltr_models"
    feedback_file: str = "data/ltr_feedback.jsonl"


@dataclass
class ClickFeedback:
    """User feedback for a search result."""
    query: str
    query_id: str
    doc_id: str
    rank: int
    clicked: bool
    dwell_time_seconds: float = 0.0
    explicit_rating: Optional[int] = None  # 1-5 scale
    timestamp: datetime = field(default_factory=datetime.utcnow)
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class RankedResult:
    """A ranked search result."""
    doc_id: str
    original_score: float
    ltr_score: float
    features: Dict[str, float]
    rank: int


# =============================================================================
# Feature Extraction
# =============================================================================

class LTRFeatureExtractor:
    """
    Extract ranking features from query-document pairs.

    Features include:
    - Lexical: BM25, term frequency, exact match
    - Semantic: Embedding similarity
    - Structural: Title match, position in document
    - Temporal: Document freshness
    - Quality: Length, readability signals
    """

    def __init__(self, config: Optional[LTRConfig] = None):
        self.config = config or LTRConfig()

    def extract(
        self,
        query: str,
        doc: Dict[str, Any],
        rank: int = 0,
    ) -> Dict[str, float]:
        """
        Extract features for a query-document pair.

        Args:
            query: Search query
            doc: Document/chunk with metadata
            rank: Original rank position

        Returns:
            Feature dictionary
        """
        features = {}

        # BM25 score (if pre-computed)
        features["bm25_score"] = doc.get("bm25_score", doc.get("score", 0.0))

        # Semantic similarity
        features["semantic_similarity"] = doc.get("similarity_score", doc.get("score", 0.0))

        # Title match
        title = doc.get("document_title", "") or doc.get("title", "")
        query_lower = query.lower()
        title_lower = title.lower()

        # Exact match in title
        features["title_match"] = 1.0 if query_lower in title_lower else 0.0

        # Partial title match (term overlap)
        query_terms = set(query_lower.split())
        title_terms = set(title_lower.split())
        if query_terms:
            features["title_term_overlap"] = len(query_terms & title_terms) / len(query_terms)
        else:
            features["title_term_overlap"] = 0.0

        # Section title match
        section = doc.get("section_title", "") or ""
        features["section_match"] = 1.0 if query_lower in section.lower() else 0.0

        # Content (for proximity calculation)
        content = doc.get("content", "")
        content_lower = content.lower()

        # Term proximity (minimum distance between query terms)
        features["term_proximity"] = self._calculate_term_proximity(query_lower, content_lower)

        # Freshness (exponential decay)
        updated_at = doc.get("updated_at") or doc.get("created_at")
        if updated_at:
            if isinstance(updated_at, str):
                try:
                    updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                except ValueError:
                    updated_at = None

            if updated_at:
                days_old = (datetime.utcnow() - updated_at.replace(tzinfo=None)).days
                features["freshness"] = np.exp(-0.01 * days_old)  # Decay rate
            else:
                features["freshness"] = 0.5
        else:
            features["freshness"] = 0.5

        # Chunk position (earlier chunks often more relevant)
        chunk_index = doc.get("chunk_index", 0)
        features["chunk_position"] = 1.0 / (1.0 + chunk_index)  # Decay with position

        # Document length (normalized)
        doc_length = len(content)
        features["doc_length"] = min(doc_length / 5000.0, 1.0)  # Normalize to ~5K chars

        # Original rank (position bias correction)
        features["original_rank"] = 1.0 / (1.0 + rank)

        return features

    def _calculate_term_proximity(self, query: str, content: str) -> float:
        """
        Calculate minimum proximity between query terms in content.

        Lower distance = higher score (terms are closer together).
        """
        query_terms = query.split()
        if len(query_terms) < 2:
            return 1.0  # Single term, max proximity

        # Find positions of each term
        positions = {}
        words = content.split()
        for i, word in enumerate(words):
            for term in query_terms:
                if term in word:
                    if term not in positions:
                        positions[term] = []
                    positions[term].append(i)

        # If not all terms found, return 0
        if len(positions) < len(query_terms):
            return 0.0

        # Find minimum span containing all terms
        min_span = float("inf")
        for term_positions in zip(*[positions.get(t, [0]) for t in query_terms]):
            span = max(term_positions) - min(term_positions)
            min_span = min(min_span, span)

        if min_span == float("inf"):
            return 0.0

        # Convert to similarity (closer = higher)
        return 1.0 / (1.0 + min_span)


# =============================================================================
# Learning-to-Rank Model
# =============================================================================

class LTRRanker:
    """
    Learning-to-Rank ranker using XGBoost.

    Trains on user feedback to learn optimal ranking weights
    that combine multiple signals (BM25, semantic, freshness, etc.).
    """

    def __init__(self, config: Optional[LTRConfig] = None):
        if not HAS_NUMPY:
            raise ImportError("NumPy required: pip install numpy")

        self.config = config or LTRConfig()
        self.feature_extractor = LTRFeatureExtractor(config)
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._feedback: List[ClickFeedback] = []
        self._is_trained: bool = False

        # Load existing feedback
        self._load_feedback()

        # Load existing model if available
        self._load_model()

    # =========================================================================
    # Feedback Collection
    # =========================================================================

    async def record_feedback(
        self,
        query: str,
        doc_id: str,
        rank: int,
        clicked: bool = False,
        dwell_time_seconds: float = 0.0,
        explicit_rating: Optional[int] = None,
        features: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record user feedback for a search result.

        Args:
            query: Search query
            doc_id: Document/chunk ID
            rank: Original rank position
            clicked: Whether user clicked
            dwell_time_seconds: Time spent on result
            explicit_rating: Optional explicit 1-5 rating
            features: Pre-computed features (if available)
        """
        import hashlib

        # Generate query ID for grouping
        query_id = hashlib.md5(query.lower().encode()).hexdigest()[:12]

        feedback = ClickFeedback(
            query=query,
            query_id=query_id,
            doc_id=doc_id,
            rank=rank,
            clicked=clicked,
            dwell_time_seconds=dwell_time_seconds,
            explicit_rating=explicit_rating,
            features=features or {},
        )

        self._feedback.append(feedback)

        # Persist feedback
        await self._save_feedback(feedback)

        logger.debug(
            "Recorded LTR feedback",
            query_id=query_id,
            doc_id=doc_id,
            clicked=clicked,
            dwell_time=dwell_time_seconds,
        )

    async def _save_feedback(self, feedback: ClickFeedback) -> None:
        """Append feedback to file."""
        os.makedirs(os.path.dirname(self.config.feedback_file), exist_ok=True)

        with open(self.config.feedback_file, "a") as f:
            record = {
                "query": feedback.query,
                "query_id": feedback.query_id,
                "doc_id": feedback.doc_id,
                "rank": feedback.rank,
                "clicked": feedback.clicked,
                "dwell_time_seconds": feedback.dwell_time_seconds,
                "explicit_rating": feedback.explicit_rating,
                "timestamp": feedback.timestamp.isoformat(),
                "features": feedback.features,
            }
            f.write(json.dumps(record) + "\n")

    def _load_feedback(self) -> None:
        """Load existing feedback from file."""
        if not os.path.exists(self.config.feedback_file):
            return

        try:
            with open(self.config.feedback_file, "r") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        feedback = ClickFeedback(
                            query=record["query"],
                            query_id=record["query_id"],
                            doc_id=record["doc_id"],
                            rank=record["rank"],
                            clicked=record["clicked"],
                            dwell_time_seconds=record.get("dwell_time_seconds", 0.0),
                            explicit_rating=record.get("explicit_rating"),
                            timestamp=datetime.fromisoformat(record["timestamp"]),
                            features=record.get("features", {}),
                        )
                        self._feedback.append(feedback)

            logger.info("Loaded LTR feedback", n_samples=len(self._feedback))

        except Exception as e:
            logger.warning("Failed to load LTR feedback", error=str(e))

    # =========================================================================
    # Training
    # =========================================================================

    def _compute_relevance_label(self, feedback: ClickFeedback) -> float:
        """
        Compute relevance label from feedback signals.

        Uses weighted combination of:
        - Click: Binary signal
        - Dwell time: Longer = more relevant
        - Explicit rating: Direct user judgment
        """
        label = 0.0

        # Click signal
        if feedback.clicked:
            label += self.config.click_weight * 1.0

        # Dwell time signal (log scale, capped at 5 minutes)
        if feedback.dwell_time_seconds > 0:
            dwell_score = np.log1p(min(feedback.dwell_time_seconds, 300)) / np.log1p(300)
            label += self.config.dwell_weight * dwell_score

        # Explicit rating (normalized to 0-1)
        if feedback.explicit_rating is not None:
            rating_score = (feedback.explicit_rating - 1) / 4.0  # 1-5 → 0-1
            label += self.config.explicit_weight * rating_score

        return label

    async def train(self, force: bool = False) -> Dict[str, Any]:
        """
        Train the LTR model on collected feedback.

        Args:
            force: Train even if minimum samples not reached

        Returns:
            Training statistics
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost required: pip install xgboost")

        n_samples = len(self._feedback)

        if not force and n_samples < self.config.min_training_samples:
            return {
                "status": "insufficient_data",
                "n_samples": n_samples,
                "min_required": self.config.min_training_samples,
            }

        logger.info("Training LTR model", n_samples=n_samples)

        # Prepare training data
        X, y, groups = self._prepare_training_data()

        if len(X) == 0:
            return {"status": "no_data", "n_samples": 0}

        # Run training in thread pool
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(
            None,
            self._train_model,
            X, y, groups,
        )

        self._is_trained = True

        # Save model
        self._save_model()

        return stats

    def _prepare_training_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features, labels, and groups for XGBoost ranking."""
        # Group feedback by query
        query_groups: Dict[str, List[ClickFeedback]] = {}
        for fb in self._feedback:
            if fb.query_id not in query_groups:
                query_groups[fb.query_id] = []
            query_groups[fb.query_id].append(fb)

        X_list = []
        y_list = []
        group_sizes = []

        for query_id, group in query_groups.items():
            if len(group) < 2:
                continue  # Need at least 2 items for pairwise

            for fb in group:
                # Extract features
                if fb.features:
                    features = [fb.features.get(name, 0.0) for name in self.config.feature_names]
                else:
                    # Use basic features if not pre-computed
                    features = [0.0] * len(self.config.feature_names)

                X_list.append(features)
                y_list.append(self._compute_relevance_label(fb))

            group_sizes.append(len(group))

        if not X_list:
            return np.array([]), np.array([]), np.array([])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        groups = np.array(group_sizes, dtype=np.int32)

        # Scale features
        if HAS_SKLEARN:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        return X, y, groups

    def _train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> Dict[str, Any]:
        """Train XGBoost ranking model."""
        # Split data
        n_groups = len(groups)
        train_size = int(n_groups * (1 - self.config.validation_split))

        # Split by groups
        train_end = sum(groups[:train_size])
        X_train, X_val = X[:train_end], X[train_end:]
        y_train, y_val = y[:train_end], y[train_end:]
        groups_train = groups[:train_size]
        groups_val = groups[train_size:]

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(groups_train)

        dval = xgb.DMatrix(X_val, label=y_val)
        dval.set_group(groups_val)

        # Training parameters
        params = {
            "objective": self.config.objective.value,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "eval_metric": "ndcg@10",
        }

        # Train
        evals_result = {}
        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=self.config.early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=False,
        )

        # Get feature importance
        importance = self._model.get_score(importance_type="gain")
        feature_importance = {
            f"f{i}": importance.get(f"f{i}", 0.0)
            for i in range(len(self.config.feature_names))
        }

        return {
            "status": "trained",
            "n_samples": len(X),
            "n_groups": n_groups,
            "best_iteration": self._model.best_iteration,
            "train_ndcg": evals_result["train"]["ndcg@10"][-1],
            "val_ndcg": evals_result["val"]["ndcg@10"][-1] if groups_val.size > 0 else None,
            "feature_importance": feature_importance,
        }

    # =========================================================================
    # Reranking
    # =========================================================================

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[RankedResult]:
        """
        Rerank candidates using the trained LTR model.

        Args:
            query: Search query
            candidates: List of candidate documents with metadata

        Returns:
            Reranked results with LTR scores
        """
        if not candidates:
            return []

        # Extract features
        features_list = []
        for i, doc in enumerate(candidates):
            features = self.feature_extractor.extract(query, doc, rank=i)
            doc["_ltr_features"] = features
            feature_vector = [features.get(name, 0.0) for name in self.config.feature_names]
            features_list.append(feature_vector)

        X = np.array(features_list, dtype=np.float32)

        # Scale if scaler available
        if self._scaler is not None:
            X = self._scaler.transform(X)

        # Get LTR scores
        if self._model is not None and self._is_trained:
            dmatrix = xgb.DMatrix(X)
            ltr_scores = self._model.predict(dmatrix)
        else:
            # Fallback: use original scores
            ltr_scores = np.array([doc.get("score", 0.0) for doc in candidates])

        # Sort by LTR score
        ranked_indices = np.argsort(ltr_scores)[::-1]

        results = []
        for rank, idx in enumerate(ranked_indices):
            doc = candidates[idx]
            results.append(RankedResult(
                doc_id=doc.get("chunk_id") or doc.get("id") or str(idx),
                original_score=doc.get("score", 0.0),
                ltr_score=float(ltr_scores[idx]),
                features=doc.get("_ltr_features", {}),
                rank=rank,
            ))

        return results

    # =========================================================================
    # Model Persistence
    # =========================================================================

    def _save_model(self) -> None:
        """Save model to disk."""
        if self._model is None:
            return

        os.makedirs(self.config.model_dir, exist_ok=True)

        model_path = os.path.join(self.config.model_dir, "ltr_model.json")
        self._model.save_model(model_path)

        # Save scaler
        if self._scaler is not None:
            import pickle
            scaler_path = os.path.join(self.config.model_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self._scaler, f)

        logger.info("Saved LTR model", path=model_path)

    def _load_model(self) -> None:
        """Load model from disk."""
        model_path = os.path.join(self.config.model_dir, "ltr_model.json")

        if not os.path.exists(model_path):
            return

        try:
            if HAS_XGBOOST:
                self._model = xgb.Booster()
                self._model.load_model(model_path)
                self._is_trained = True

                # Load scaler
                scaler_path = os.path.join(self.config.model_dir, "scaler.pkl")
                if os.path.exists(scaler_path):
                    import pickle
                    with open(scaler_path, "rb") as f:
                        self._scaler = pickle.load(f)

                logger.info("Loaded LTR model", path=model_path)

        except Exception as e:
            logger.warning("Failed to load LTR model", error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Get LTR statistics."""
        return {
            "is_trained": self._is_trained,
            "n_feedback_samples": len(self._feedback),
            "min_training_samples": self.config.min_training_samples,
            "xgboost_available": HAS_XGBOOST,
            "feature_names": self.config.feature_names,
        }


# =============================================================================
# Singleton
# =============================================================================

_ltr_ranker: Optional[LTRRanker] = None


def get_ltr_ranker(
    config: Optional[LTRConfig] = None,
) -> LTRRanker:
    """Get or create LTR ranker singleton."""
    global _ltr_ranker

    if _ltr_ranker is None or config is not None:
        _ltr_ranker = LTRRanker(config)

    return _ltr_ranker
