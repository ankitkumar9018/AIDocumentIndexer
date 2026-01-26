"""
AIDocumentIndexer - DSPy Example Collector
==========================================

Phase 93: Mines training examples from existing user interaction data
for DSPy prompt optimization.

Data Sources:
1. ChatFeedback (rating >= 4) → RAG answer examples
2. AgentTrajectory (quality >= 3.5) → Agentic RAG examples
3. DSPyTrainingExample table → Manually curated examples
"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple

import structlog
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = structlog.get_logger(__name__)


class DSPyExampleCollector:
    """
    Collects and formats training examples from database sources
    for DSPy prompt optimization.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def collect_rag_examples(
        self,
        min_rating: int = 4,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Collect RAG answer examples from chat feedback.

        Queries ChatMessage + ChatFeedback for highly-rated interactions.

        Args:
            min_rating: Minimum feedback rating (1-5)
            limit: Maximum examples to collect

        Returns:
            List of example dicts with context, question, answer keys
        """
        from backend.db.models import ChatMessage, ChatFeedback

        try:
            # Get messages with positive feedback
            result = await self.db.execute(
                select(ChatMessage, ChatFeedback)
                .join(
                    ChatFeedback,
                    ChatFeedback.message_id == ChatMessage.id,
                    isouter=True,
                )
                .where(
                    and_(
                        ChatMessage.role == "assistant",
                        ChatFeedback.rating >= min_rating,
                    )
                )
                .order_by(ChatFeedback.rating.desc())
                .limit(limit)
            )
            rows = result.all()

            examples = []
            for message, feedback in rows:
                # Get the preceding user message for the question
                user_msg_result = await self.db.execute(
                    select(ChatMessage)
                    .where(
                        and_(
                            ChatMessage.session_id == message.session_id,
                            ChatMessage.role == "user",
                            ChatMessage.created_at < message.created_at,
                        )
                    )
                    .order_by(ChatMessage.created_at.desc())
                    .limit(1)
                )
                user_msg = user_msg_result.scalar_one_or_none()

                if user_msg and message.content:
                    # Extract source context from message metadata
                    source_chunks = ""
                    if hasattr(message, 'source_chunks') and message.source_chunks:
                        try:
                            chunks = json.loads(message.source_chunks) if isinstance(
                                message.source_chunks, str
                            ) else message.source_chunks
                            source_chunks = "\n\n".join(
                                c.get("content", "") for c in chunks[:5]
                            )
                        except (json.JSONDecodeError, TypeError):
                            pass

                    examples.append({
                        "context": source_chunks or "No context available",
                        "question": user_msg.content,
                        "answer": message.content,
                        "suggested_questions": "",
                        "source": "chat_feedback",
                        "quality_score": (feedback.rating / 5.0) if feedback else 0.8,
                    })

            logger.info(
                "Collected RAG examples from chat feedback",
                count=len(examples),
                min_rating=min_rating,
            )
            return examples

        except Exception as e:
            logger.warning("Failed to collect chat feedback examples", error=str(e))
            return []

    async def collect_trajectory_examples(
        self,
        min_quality: float = 3.5,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Collect agentic RAG examples from agent trajectories.

        Args:
            min_quality: Minimum quality score (0-5)
            limit: Maximum examples

        Returns:
            List of example dicts for ReAct reasoning signature
        """
        from backend.db.models import AgentTrajectory

        try:
            result = await self.db.execute(
                select(AgentTrajectory)
                .where(
                    and_(
                        AgentTrajectory.success == True,
                        AgentTrajectory.quality_score >= min_quality,
                    )
                )
                .order_by(AgentTrajectory.quality_score.desc())
                .limit(limit)
            )
            trajectories = result.scalars().all()

            examples = []
            for traj in trajectories:
                steps = traj.trajectory_steps or []
                if not steps:
                    continue

                # Extract query from first step
                query = ""
                if isinstance(steps, list) and len(steps) > 0:
                    first_step = steps[0] if isinstance(steps[0], dict) else {}
                    query = first_step.get("input", first_step.get("query", ""))

                if not query:
                    continue

                # Build examples from trajectory steps
                for i, step in enumerate(steps):
                    if not isinstance(step, dict):
                        continue

                    previous_steps = json.dumps(steps[:i]) if i > 0 else ""
                    current_knowledge = step.get("observation", step.get("result", ""))

                    examples.append({
                        "query": query,
                        "previous_steps": previous_steps,
                        "current_knowledge": current_knowledge,
                        "thought": step.get("thought", ""),
                        "action": step.get("action", "search"),
                        "action_input": step.get("action_input", ""),
                        "source": "trajectory",
                        "quality_score": (traj.quality_score or 3.5) / 5.0,
                    })

            logger.info(
                "Collected trajectory examples",
                count=len(examples),
                min_quality=min_quality,
            )
            return examples

        except Exception as e:
            logger.warning("Failed to collect trajectory examples", error=str(e))
            return []

    async def collect_manual_examples(
        self,
        signature_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect manually curated training examples.

        Args:
            signature_name: Optional filter by signature type

        Returns:
            List of example dicts
        """
        try:
            from backend.db.models import DSPyTrainingExample

            query = select(DSPyTrainingExample).where(
                DSPyTrainingExample.is_active == True
            )
            if signature_name:
                query = query.where(
                    DSPyTrainingExample.signature_name == signature_name
                )

            result = await self.db.execute(query)
            rows = result.scalars().all()

            examples = []
            for row in rows:
                example = {
                    **(row.inputs or {}),
                    **(row.outputs or {}),
                    "source": "manual",
                    "quality_score": row.quality_score,
                }
                examples.append(example)

            logger.info(
                "Collected manual examples",
                count=len(examples),
                signature=signature_name,
            )
            return examples

        except Exception as e:
            logger.warning("Failed to collect manual examples", error=str(e))
            return []

    async def build_dataset(
        self,
        signature_name: str,
        train_ratio: float = 0.8,
        max_examples: int = 500,
    ) -> Tuple[List, List]:
        """
        Build train/dev datasets for a specific DSPy signature.

        Combines examples from all sources, converts to dspy.Example format,
        and splits into train/dev sets.

        Args:
            signature_name: Which signature to build dataset for
            train_ratio: Fraction of data for training (rest is dev)
            max_examples: Maximum total examples

        Returns:
            Tuple of (train_examples, dev_examples) as dspy.Example lists
        """
        if not DSPY_AVAILABLE:
            raise ImportError("dspy-ai is required for dataset building")

        all_examples = []

        # Collect from appropriate sources based on signature
        if signature_name in ("rag_answer", "answer_synthesis"):
            rag_examples = await self.collect_rag_examples(limit=max_examples)
            all_examples.extend(rag_examples)

        if signature_name in ("react_reasoning", "query_decomposition", "agentic_rag"):
            traj_examples = await self.collect_trajectory_examples(limit=max_examples)
            all_examples.extend(traj_examples)

        # Always include manual examples
        manual_examples = await self.collect_manual_examples(
            signature_name=signature_name
        )
        all_examples.extend(manual_examples)

        if not all_examples:
            logger.warning(
                "No training examples found",
                signature=signature_name,
            )
            return [], []

        # Sort by quality and limit
        all_examples.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        all_examples = all_examples[:max_examples]

        # Convert to dspy.Example
        dspy_examples = []
        for ex in all_examples:
            # Remove metadata fields
            clean = {
                k: v for k, v in ex.items()
                if k not in ("source", "quality_score")
            }
            dspy_examples.append(dspy.Example(**clean).with_inputs(
                *self._get_input_fields(signature_name)
            ))

        # Shuffle and split
        random.shuffle(dspy_examples)
        split_idx = int(len(dspy_examples) * train_ratio)
        train = dspy_examples[:split_idx]
        dev = dspy_examples[split_idx:]

        logger.info(
            "Built DSPy dataset",
            signature=signature_name,
            total=len(dspy_examples),
            train=len(train),
            dev=len(dev),
        )

        return train, dev

    @staticmethod
    def _get_input_fields(signature_name: str) -> List[str]:
        """Get input field names for a signature."""
        input_fields = {
            "rag_answer": ["context", "question"],
            "query_expansion": ["original_query", "num_variations"],
            "query_decomposition": ["query"],
            "react_reasoning": ["query", "previous_steps", "current_knowledge"],
            "answer_synthesis": ["query", "sub_answers", "graph_context", "retrieved_context"],
        }
        return input_fields.get(signature_name, [])

    async def get_example_count(self, signature_name: str) -> Dict[str, int]:
        """Get count of available examples by source."""
        counts = {
            "chat_feedback": 0,
            "trajectory": 0,
            "manual": 0,
            "total": 0,
        }

        try:
            from backend.db.models import ChatFeedback
            result = await self.db.execute(
                select(func.count(ChatFeedback.id)).where(
                    ChatFeedback.rating >= 4
                )
            )
            counts["chat_feedback"] = result.scalar() or 0
        except Exception:
            pass

        try:
            from backend.db.models import AgentTrajectory
            result = await self.db.execute(
                select(func.count(AgentTrajectory.id)).where(
                    and_(
                        AgentTrajectory.success == True,
                        AgentTrajectory.quality_score >= 3.5,
                    )
                )
            )
            counts["trajectory"] = result.scalar() or 0
        except Exception:
            pass

        try:
            from backend.db.models import DSPyTrainingExample
            query = select(func.count(DSPyTrainingExample.id)).where(
                DSPyTrainingExample.is_active == True
            )
            if signature_name:
                query = query.where(
                    DSPyTrainingExample.signature_name == signature_name
                )
            result = await self.db.execute(query)
            counts["manual"] = result.scalar() or 0
        except Exception:
            pass

        counts["total"] = sum(v for k, v in counts.items() if k != "total")
        return counts
