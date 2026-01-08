"""
AIDocumentIndexer - Trajectory Collector Service
================================================

Records agent execution trajectories for:
- Performance analysis
- Self-improvement (Prompt Builder)
- Debugging and auditing
- Quality metrics tracking

Trajectories capture:
- All reasoning steps
- Tool calls and results
- LLM invocations
- Success/failure outcomes
- Token usage and timing
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import structlog
from sqlalchemy import select, func, and_, desc, or_, Integer, case
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import AgentTrajectory, AgentDefinition, AgentPromptVersion
from backend.services.agents.agent_base import TrajectoryStep, TaskStatus

logger = structlog.get_logger(__name__)


class TrajectoryCollector:
    """
    Collects and persists agent execution trajectories.

    Trajectories are used by the Prompt Builder Agent for self-improvement
    and by administrators for debugging and performance analysis.
    """

    def __init__(self, db: Optional[AsyncSession] = None):
        """
        Initialize collector.

        Args:
            db: Optional database session (can be set later via set_db)
        """
        self.db = db

    def set_db(self, db: AsyncSession) -> None:
        """Set database session."""
        self.db = db

    async def record_trajectory_by_type(
        self,
        session_id: str,
        agent_type: str,
        task_type: str,
        input_summary: str,
        steps: List[TrajectoryStep],
        success: bool,
        quality_score: Optional[float] = None,
        error_message: Optional[str] = None,
        total_tokens: int = 0,
        total_duration_ms: int = 0,
        total_cost_usd: float = 0.0,
    ) -> Optional[AgentTrajectory]:
        """
        Record trajectory by looking up AgentDefinition by agent_type.

        This is the preferred method for recording trajectories as it handles
        the mapping from agent_type string to AgentDefinition UUID.

        Args:
            session_id: Execution session ID
            agent_type: Agent type string (e.g., "research", "generator", "manager")
            task_type: Type of task executed
            input_summary: Summary of task inputs
            steps: List of execution steps
            success: Whether execution succeeded
            quality_score: Quality score (0-5)
            error_message: Error message if failed
            total_tokens: Total tokens used
            total_duration_ms: Total duration in milliseconds
            total_cost_usd: Total cost in USD

        Returns:
            Created AgentTrajectory record or None
        """
        if not self.db:
            logger.warning("No database session, trajectory not persisted")
            return None

        # Look up AgentDefinition by agent_type
        result = await self.db.execute(
            select(AgentDefinition).where(AgentDefinition.agent_type == agent_type)
        )
        agent_def = result.scalar_one_or_none()

        if not agent_def:
            logger.warning(f"No AgentDefinition found for type: {agent_type}")

        # Convert steps to JSON-serializable format
        trajectory_steps = []
        calc_tokens = 0
        calc_duration = 0

        for step in steps:
            step_dict = {
                "step_id": step.step_id,
                "timestamp": step.timestamp.isoformat(),
                "agent_id": step.agent_id,
                "action_type": step.action_type,
                "input_data": step.input_data,
                "output_data": step.output_data,
                "tokens_used": step.tokens_used,
                "duration_ms": step.duration_ms,
                "success": step.success,
                "error_message": step.error_message,
            }
            trajectory_steps.append(step_dict)
            calc_tokens += step.tokens_used
            calc_duration += step.duration_ms

        # Use provided totals or calculate from steps
        final_tokens = total_tokens if total_tokens > 0 else calc_tokens
        final_duration = total_duration_ms if total_duration_ms > 0 else calc_duration

        # Create trajectory record
        trajectory = AgentTrajectory(
            session_id=uuid.UUID(session_id) if isinstance(session_id, str) else session_id,
            agent_id=agent_def.id if agent_def else None,
            task_type=task_type,
            input_summary=input_summary[:1000] if input_summary else None,
            trajectory_steps=trajectory_steps,
            success=success,
            quality_score=quality_score,
            error_message=error_message,
            total_tokens=final_tokens,
            total_duration_ms=final_duration,
            total_cost_usd=total_cost_usd,
        )

        self.db.add(trajectory)

        # Update agent metrics after recording
        if agent_def:
            await self._update_agent_metrics(agent_def)

        await self.db.commit()
        await self.db.refresh(trajectory)

        logger.info(
            "Recorded trajectory by type",
            trajectory_id=str(trajectory.id),
            agent_type=agent_type,
            agent_id=str(agent_def.id) if agent_def else None,
            success=success,
            total_tokens=final_tokens,
        )

        return trajectory

    async def _update_agent_metrics(self, agent_def: AgentDefinition) -> None:
        """
        Update AgentDefinition metrics after recording trajectory.

        This keeps the success_rate, total_executions, avg_latency_ms, and
        avg_tokens_per_execution fields up to date.
        """
        if not self.db or not agent_def:
            return

        # Get stats for this agent from recent trajectories (last 7 days)
        cutoff = datetime.utcnow() - timedelta(days=7)

        # Count total executions and successes
        stats = await self.db.execute(
            select(
                func.count(AgentTrajectory.id).label("total"),
                func.sum(case((AgentTrajectory.success == True, 1), else_=0)).label("successes"),
                func.avg(AgentTrajectory.total_duration_ms).label("avg_latency"),
                func.avg(AgentTrajectory.total_tokens).label("avg_tokens"),
            )
            .where(and_(
                AgentTrajectory.agent_id == agent_def.id,
                AgentTrajectory.created_at >= cutoff,
            ))
        )
        row = stats.one()

        # Update agent definition metrics
        agent_def.total_executions = row.total or 0
        agent_def.success_rate = (row.successes / row.total) if row.total and row.total > 0 else 0.0
        agent_def.avg_latency_ms = int(row.avg_latency) if row.avg_latency else None
        agent_def.avg_tokens_per_execution = int(row.avg_tokens) if row.avg_tokens else None

        logger.debug(
            "Updated agent metrics",
            agent_id=str(agent_def.id),
            agent_type=agent_def.agent_type,
            total_executions=agent_def.total_executions,
            success_rate=agent_def.success_rate,
        )

    async def record_trajectory(
        self,
        session_id: str,
        agent_id: str,
        task_type: str,
        input_summary: str,
        steps: List[TrajectoryStep],
        success: bool,
        quality_score: Optional[float] = None,
        error_message: Optional[str] = None,
        prompt_version_id: Optional[str] = None,
        user_rating: Optional[int] = None,
        user_feedback: Optional[str] = None,
    ) -> AgentTrajectory:
        """
        Record a complete trajectory to the database.

        Args:
            session_id: Execution session ID
            agent_id: Agent that executed the task
            task_type: Type of task executed
            input_summary: Summary of task inputs
            steps: List of execution steps
            success: Whether execution succeeded
            quality_score: Quality score (0-5)
            error_message: Error message if failed
            prompt_version_id: Prompt version used
            user_rating: Optional user rating (1-5)
            user_feedback: Optional user feedback text

        Returns:
            Created AgentTrajectory record
        """
        if not self.db:
            logger.warning("No database session, trajectory not persisted")
            return None

        # Convert steps to JSON-serializable format
        trajectory_steps = []
        total_tokens = 0
        total_duration_ms = 0

        for step in steps:
            step_dict = {
                "step_id": step.step_id,
                "timestamp": step.timestamp.isoformat(),
                "agent_id": step.agent_id,
                "action_type": step.action_type,
                "input_data": step.input_data,
                "output_data": step.output_data,
                "tokens_used": step.tokens_used,
                "duration_ms": step.duration_ms,
                "success": step.success,
                "error_message": step.error_message,
            }
            trajectory_steps.append(step_dict)
            total_tokens += step.tokens_used
            total_duration_ms += step.duration_ms

        # Create trajectory record
        trajectory = AgentTrajectory(
            session_id=uuid.UUID(session_id) if isinstance(session_id, str) else session_id,
            agent_id=uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id,
            task_type=task_type,
            input_summary=input_summary[:1000] if input_summary else None,  # Limit length
            trajectory_steps=trajectory_steps,
            success=success,
            quality_score=quality_score,
            error_message=error_message,
            total_tokens=total_tokens,
            total_duration_ms=total_duration_ms,
            prompt_version_id=uuid.UUID(prompt_version_id) if prompt_version_id else None,
            user_rating=user_rating,
            user_feedback=user_feedback,
        )

        self.db.add(trajectory)
        await self.db.commit()
        await self.db.refresh(trajectory)

        logger.info(
            "Recorded trajectory",
            trajectory_id=str(trajectory.id),
            agent_id=agent_id,
            task_type=task_type,
            success=success,
            total_tokens=total_tokens,
        )

        return trajectory

    async def get_trajectory(
        self,
        trajectory_id: str
    ) -> Optional[AgentTrajectory]:
        """
        Get a specific trajectory by ID.

        Args:
            trajectory_id: Trajectory UUID

        Returns:
            AgentTrajectory or None
        """
        if not self.db:
            return None

        result = await self.db.execute(
            select(AgentTrajectory).where(
                AgentTrajectory.id == uuid.UUID(trajectory_id)
            )
        )
        return result.scalar_one_or_none()

    async def get_session_trajectories(
        self,
        session_id: str
    ) -> List[AgentTrajectory]:
        """
        Get all trajectories for a session.

        Args:
            session_id: Session UUID

        Returns:
            List of trajectories
        """
        if not self.db:
            return []

        result = await self.db.execute(
            select(AgentTrajectory)
            .where(AgentTrajectory.session_id == uuid.UUID(session_id))
            .order_by(AgentTrajectory.created_at)
        )
        return list(result.scalars().all())

    async def get_recent_trajectories(
        self,
        agent_id: Optional[str] = None,
        hours: int = 24,
        success_filter: Optional[bool] = None,
        limit: int = 100,
    ) -> List[AgentTrajectory]:
        """
        Get recent trajectories, optionally filtered.

        Args:
            agent_id: Optional filter by agent
            hours: Time window in hours
            success_filter: Optional filter by success status
            limit: Maximum results

        Returns:
            List of trajectories
        """
        if not self.db:
            return []

        cutoff = datetime.utcnow() - timedelta(hours=hours)

        conditions = [AgentTrajectory.created_at >= cutoff]

        if agent_id:
            conditions.append(AgentTrajectory.agent_id == uuid.UUID(agent_id))

        if success_filter is not None:
            conditions.append(AgentTrajectory.success == success_filter)

        result = await self.db.execute(
            select(AgentTrajectory)
            .where(and_(*conditions))
            .order_by(desc(AgentTrajectory.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_failed_trajectories(
        self,
        agent_id: str,
        hours: int = 24,
        limit: int = 50,
    ) -> List[AgentTrajectory]:
        """
        Get failed trajectories for an agent within time window.

        Used by Prompt Builder for failure analysis.

        Args:
            agent_id: Agent UUID
            hours: Time window in hours
            limit: Maximum results

        Returns:
            List of failed trajectories
        """
        return await self.get_recent_trajectories(
            agent_id=agent_id,
            hours=hours,
            success_filter=False,
            limit=limit,
        )

    async def get_success_rate(
        self,
        agent_id: str,
        hours: int = 24
    ) -> float:
        """
        Calculate success rate for an agent.

        Args:
            agent_id: Agent UUID
            hours: Time window in hours

        Returns:
            Success rate as float (0.0 to 1.0)
        """
        if not self.db:
            return 0.0

        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Count total and successful
        total_result = await self.db.execute(
            select(func.count(AgentTrajectory.id))
            .where(and_(
                AgentTrajectory.agent_id == uuid.UUID(agent_id),
                AgentTrajectory.created_at >= cutoff,
            ))
        )
        total = total_result.scalar() or 0

        if total == 0:
            return 0.0

        success_result = await self.db.execute(
            select(func.count(AgentTrajectory.id))
            .where(and_(
                AgentTrajectory.agent_id == uuid.UUID(agent_id),
                AgentTrajectory.created_at >= cutoff,
                AgentTrajectory.success == True,
            ))
        )
        successes = success_result.scalar() or 0

        return successes / total

    async def get_avg_quality_score(
        self,
        agent_id: str,
        hours: int = 24
    ) -> Optional[float]:
        """
        Calculate average quality score for an agent.

        Args:
            agent_id: Agent UUID
            hours: Time window in hours

        Returns:
            Average quality score or None if no data
        """
        if not self.db:
            return None

        cutoff = datetime.utcnow() - timedelta(hours=hours)

        result = await self.db.execute(
            select(func.avg(AgentTrajectory.quality_score))
            .where(and_(
                AgentTrajectory.agent_id == uuid.UUID(agent_id),
                AgentTrajectory.created_at >= cutoff,
                AgentTrajectory.quality_score.isnot(None),
            ))
        )
        return result.scalar()

    async def get_agent_stats(
        self,
        agent_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get comprehensive stats for an agent.

        Args:
            agent_id: Agent UUID
            hours: Time window in hours

        Returns:
            Dict with success_rate, avg_quality, total_executions, etc.
        """
        if not self.db:
            return {}

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        agent_uuid = uuid.UUID(agent_id)

        # Get aggregate stats
        # Use case() for boolean to int conversion - works across SQLite and PostgreSQL
        stats_result = await self.db.execute(
            select(
                func.count(AgentTrajectory.id).label("total"),
                func.sum(
                    case((AgentTrajectory.success == True, 1), else_=0)
                ).label("successes"),
                func.avg(AgentTrajectory.quality_score).label("avg_quality"),
                func.avg(AgentTrajectory.total_tokens).label("avg_tokens"),
                func.avg(AgentTrajectory.total_duration_ms).label("avg_duration"),
                func.sum(AgentTrajectory.total_tokens).label("total_tokens"),
                func.sum(AgentTrajectory.total_cost_usd).label("total_cost"),
            )
            .where(and_(
                AgentTrajectory.agent_id == agent_uuid,
                AgentTrajectory.created_at >= cutoff,
            ))
        )
        row = stats_result.one()

        total = row.total or 0
        successes = row.successes or 0
        success_rate = (successes / total) if total > 0 else 0.0

        return {
            "agent_id": agent_id,
            "time_window_hours": hours,
            "total_executions": total,
            "successful_executions": successes,
            "success_rate": success_rate,
            "avg_quality_score": float(row.avg_quality) if row.avg_quality else None,
            "avg_tokens_per_execution": int(row.avg_tokens) if row.avg_tokens else 0,
            "avg_duration_ms": int(row.avg_duration) if row.avg_duration else 0,
            "total_tokens_used": int(row.total_tokens) if row.total_tokens else 0,
            "total_cost_usd": float(row.total_cost) if row.total_cost else 0.0,
        }

    async def get_prompt_version_stats(
        self,
        prompt_version_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get stats for a specific prompt version (for A/B testing).

        Args:
            prompt_version_id: Prompt version UUID
            hours: Time window in hours

        Returns:
            Dict with execution count, success rate, avg quality
        """
        if not self.db:
            return {}

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        version_uuid = uuid.UUID(prompt_version_id)

        stats_result = await self.db.execute(
            select(
                func.count(AgentTrajectory.id).label("total"),
                func.sum(
                    case((AgentTrajectory.success == True, 1), else_=0)
                ).label("successes"),
                func.avg(AgentTrajectory.quality_score).label("avg_quality"),
            )
            .where(and_(
                AgentTrajectory.prompt_version_id == version_uuid,
                AgentTrajectory.created_at >= cutoff,
            ))
        )
        row = stats_result.one()

        total = row.total or 0
        successes = row.successes or 0

        return {
            "prompt_version_id": prompt_version_id,
            "execution_count": total,
            "success_count": successes,
            "success_rate": (successes / total) if total > 0 else 0.0,
            "avg_quality_score": float(row.avg_quality) if row.avg_quality else None,
        }

    async def add_user_feedback(
        self,
        trajectory_id: str,
        rating: int,
        feedback: Optional[str] = None
    ) -> Optional[AgentTrajectory]:
        """
        Add user feedback to a trajectory.

        Args:
            trajectory_id: Trajectory UUID
            rating: User rating (1-5)
            feedback: Optional feedback text

        Returns:
            Updated trajectory or None
        """
        if not self.db:
            return None

        trajectory = await self.get_trajectory(trajectory_id)
        if not trajectory:
            return None

        trajectory.user_rating = rating
        trajectory.user_feedback = feedback

        await self.db.commit()
        await self.db.refresh(trajectory)

        logger.info(
            "Added user feedback to trajectory",
            trajectory_id=trajectory_id,
            rating=rating,
        )

        return trajectory

    async def update_quality_score(
        self,
        trajectory_id: str,
        quality_score: float
    ) -> Optional[AgentTrajectory]:
        """
        Update quality score for a trajectory (e.g., from Critic agent).

        Args:
            trajectory_id: Trajectory UUID
            quality_score: Quality score (0-5)

        Returns:
            Updated trajectory or None
        """
        if not self.db:
            return None

        trajectory = await self.get_trajectory(trajectory_id)
        if not trajectory:
            return None

        trajectory.quality_score = quality_score

        await self.db.commit()
        await self.db.refresh(trajectory)

        return trajectory

    async def cleanup_old_trajectories(
        self,
        days: int = 30
    ) -> int:
        """
        Delete trajectories older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of deleted records
        """
        if not self.db:
            return 0

        cutoff = datetime.utcnow() - timedelta(days=days)

        # Count before delete
        count_result = await self.db.execute(
            select(func.count(AgentTrajectory.id))
            .where(AgentTrajectory.created_at < cutoff)
        )
        count = count_result.scalar() or 0

        if count > 0:
            from sqlalchemy import delete
            await self.db.execute(
                delete(AgentTrajectory)
                .where(AgentTrajectory.created_at < cutoff)
            )
            await self.db.commit()

            logger.info(
                "Cleaned up old trajectories",
                deleted_count=count,
                older_than_days=days,
            )

        return count


# Context manager for trajectory recording
class TrajectoryContext:
    """
    Context manager for recording trajectories.

    Usage:
        async with TrajectoryContext(collector, session_id, agent_id) as ctx:
            ctx.record_step(...)
            ...
        # Trajectory automatically saved on exit
    """

    def __init__(
        self,
        collector: TrajectoryCollector,
        session_id: str,
        agent_id: str,
        task_type: str,
        input_summary: str = "",
        prompt_version_id: Optional[str] = None,
    ):
        self.collector = collector
        self.session_id = session_id
        self.agent_id = agent_id
        self.task_type = task_type
        self.input_summary = input_summary
        self.prompt_version_id = prompt_version_id
        self.steps: List[TrajectoryStep] = []
        self.success = True
        self.error_message: Optional[str] = None
        self.quality_score: Optional[float] = None

    async def __aenter__(self) -> "TrajectoryContext":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val)

        await self.collector.record_trajectory(
            session_id=self.session_id,
            agent_id=self.agent_id,
            task_type=self.task_type,
            input_summary=self.input_summary,
            steps=self.steps,
            success=self.success,
            quality_score=self.quality_score,
            error_message=self.error_message,
            prompt_version_id=self.prompt_version_id,
        )

        return False  # Don't suppress exceptions

    def record_step(
        self,
        action_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        tokens_used: int = 0,
        duration_ms: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> TrajectoryStep:
        """Record a step in the trajectory."""
        step = TrajectoryStep(
            step_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            agent_id=self.agent_id,
            action_type=action_type,
            input_data=input_data,
            output_data=output_data,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
        )
        self.steps.append(step)

        if not success:
            self.success = False
            if error_message:
                self.error_message = error_message

        return step

    def set_quality_score(self, score: float) -> None:
        """Set the quality score for the trajectory."""
        self.quality_score = score

    def mark_failed(self, error: str) -> None:
        """Mark the trajectory as failed."""
        self.success = False
        self.error_message = error


# =============================================================================
# Adaptive Planning from Trajectory History
# =============================================================================

@dataclass
class PlanningHints:
    """Hints for planning based on historical trajectory analysis."""
    recommended_steps: List[str] = field(default_factory=list)
    common_patterns: List[str] = field(default_factory=list)
    avg_tokens: int = 0
    avg_duration_ms: int = 0
    success_rate: float = 0.0
    common_failures: List[str] = field(default_factory=list)
    optimization_tips: List[str] = field(default_factory=list)
    similar_task_count: int = 0


@dataclass
class TaskPattern:
    """Pattern extracted from successful trajectories."""
    task_type: str
    common_steps: List[str]
    avg_step_count: float
    success_rate: float
    sample_count: int


class AdaptivePlanner:
    """
    Learns from past trajectories to improve future planning.

    Analyzes successful executions to:
    - Identify common patterns and steps
    - Estimate resource requirements
    - Avoid common failure modes
    - Provide optimization suggestions
    """

    def __init__(self, collector: TrajectoryCollector):
        """
        Initialize adaptive planner.

        Args:
            collector: Trajectory collector for database access
        """
        self.collector = collector
        self._pattern_cache: Dict[str, TaskPattern] = {}
        self._cache_ttl_minutes = 30
        self._cache_updated_at: Optional[datetime] = None

    async def get_planning_hints(
        self,
        task_type: str,
        input_summary: Optional[str] = None,
        agent_type: Optional[str] = None,
    ) -> PlanningHints:
        """
        Get planning hints based on historical trajectories.

        Args:
            task_type: Type of task being planned
            input_summary: Optional task description for similarity matching
            agent_type: Optional agent type filter

        Returns:
            PlanningHints with recommendations
        """
        if not self.collector.db:
            return PlanningHints()

        hints = PlanningHints()

        # Get successful trajectories of similar type
        similar_trajectories = await self._get_similar_trajectories(
            task_type=task_type,
            agent_type=agent_type,
            min_quality_score=0.7,
            limit=50,
        )

        if not similar_trajectories:
            return hints

        hints.similar_task_count = len(similar_trajectories)

        # Extract patterns from successful trajectories
        hints.recommended_steps = await self._extract_common_steps(similar_trajectories)
        hints.common_patterns = await self._extract_patterns(similar_trajectories)

        # Calculate averages
        total_tokens = sum(t.total_tokens or 0 for t in similar_trajectories)
        total_duration = sum(t.total_duration_ms or 0 for t in similar_trajectories)
        success_count = sum(1 for t in similar_trajectories if t.success)

        hints.avg_tokens = total_tokens // len(similar_trajectories) if similar_trajectories else 0
        hints.avg_duration_ms = total_duration // len(similar_trajectories) if similar_trajectories else 0
        hints.success_rate = success_count / len(similar_trajectories) if similar_trajectories else 0.0

        # Get failure patterns
        hints.common_failures = await self._extract_failure_patterns(
            task_type=task_type,
            agent_type=agent_type,
        )

        # Generate optimization tips
        hints.optimization_tips = self._generate_optimization_tips(hints)

        logger.info(
            "Generated planning hints",
            task_type=task_type,
            similar_tasks=hints.similar_task_count,
            success_rate=hints.success_rate,
            recommended_steps=len(hints.recommended_steps),
        )

        return hints

    async def _get_similar_trajectories(
        self,
        task_type: str,
        agent_type: Optional[str] = None,
        min_quality_score: float = 0.5,
        limit: int = 50,
    ) -> List[AgentTrajectory]:
        """Get trajectories similar to the given task."""
        if not self.collector.db:
            return []

        conditions = [
            AgentTrajectory.task_type == task_type,
            AgentTrajectory.success == True,
        ]

        if min_quality_score > 0:
            conditions.append(
                or_(
                    AgentTrajectory.quality_score >= min_quality_score,
                    AgentTrajectory.quality_score.is_(None),
                )
            )

        # Filter by agent type if provided
        if agent_type:
            # Join with AgentDefinition to filter by type
            result = await self.collector.db.execute(
                select(AgentTrajectory)
                .join(AgentDefinition, AgentTrajectory.agent_id == AgentDefinition.id)
                .where(
                    and_(
                        *conditions,
                        AgentDefinition.agent_type == agent_type,
                    )
                )
                .order_by(desc(AgentTrajectory.created_at))
                .limit(limit)
            )
        else:
            result = await self.collector.db.execute(
                select(AgentTrajectory)
                .where(and_(*conditions))
                .order_by(desc(AgentTrajectory.created_at))
                .limit(limit)
            )

        return list(result.scalars().all())

    async def _extract_common_steps(
        self,
        trajectories: List[AgentTrajectory],
    ) -> List[str]:
        """Extract common step patterns from trajectories."""
        step_counts: Dict[str, int] = defaultdict(int)

        for trajectory in trajectories:
            if not trajectory.trajectory_steps:
                continue

            # Extract action types from steps
            seen_actions = set()
            for step in trajectory.trajectory_steps:
                action = step.get("action_type", "unknown")
                if action not in seen_actions:
                    step_counts[action] += 1
                    seen_actions.add(action)

        # Return steps that appear in >50% of trajectories
        threshold = len(trajectories) * 0.5
        common_steps = [
            action for action, count in step_counts.items()
            if count >= threshold
        ]

        # Sort by frequency
        common_steps.sort(key=lambda x: step_counts[x], reverse=True)

        return common_steps[:10]

    async def _extract_patterns(
        self,
        trajectories: List[AgentTrajectory],
    ) -> List[str]:
        """Extract high-level patterns from trajectories."""
        patterns = []

        if not trajectories:
            return patterns

        # Analyze step sequences
        step_sequences: List[List[str]] = []
        for trajectory in trajectories:
            if trajectory.trajectory_steps:
                sequence = [
                    step.get("action_type", "unknown")
                    for step in trajectory.trajectory_steps
                ]
                step_sequences.append(sequence)

        if step_sequences:
            # Find common starting patterns
            start_patterns: Dict[str, int] = defaultdict(int)
            for seq in step_sequences:
                if len(seq) >= 2:
                    pattern = f"{seq[0]} → {seq[1]}"
                    start_patterns[pattern] += 1

            # Add common start patterns
            for pattern, count in start_patterns.items():
                if count >= len(step_sequences) * 0.3:
                    patterns.append(f"Common start: {pattern}")

            # Analyze step counts
            avg_steps = sum(len(s) for s in step_sequences) / len(step_sequences)
            patterns.append(f"Average {avg_steps:.1f} steps per execution")

        return patterns

    async def _extract_failure_patterns(
        self,
        task_type: str,
        agent_type: Optional[str] = None,
    ) -> List[str]:
        """Extract common failure patterns."""
        if not self.collector.db:
            return []

        conditions = [
            AgentTrajectory.task_type == task_type,
            AgentTrajectory.success == False,
            AgentTrajectory.error_message.isnot(None),
        ]

        result = await self.collector.db.execute(
            select(AgentTrajectory.error_message)
            .where(and_(*conditions))
            .limit(20)
        )
        errors = [row[0] for row in result.all() if row[0]]

        # Count error patterns
        error_counts: Dict[str, int] = defaultdict(int)
        for error in errors:
            # Normalize error message
            error_key = error[:100].lower()
            error_counts[error_key] += 1

        # Return most common errors
        common_errors = sorted(
            error_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return [f"({count}x) {error}" for error, count in common_errors]

    def _generate_optimization_tips(self, hints: PlanningHints) -> List[str]:
        """Generate optimization tips based on analysis."""
        tips = []

        if hints.success_rate < 0.7:
            tips.append("Success rate below 70% - consider adding validation steps")

        if hints.avg_tokens > 10000:
            tips.append("High token usage - consider chunking or summarization")

        if hints.avg_duration_ms > 30000:
            tips.append("Long execution time - consider parallel processing")

        if hints.common_failures:
            tips.append(f"Watch for common failures: {hints.common_failures[0][:50]}")

        if "research" in hints.recommended_steps and "generator" in hints.recommended_steps:
            tips.append("Pattern suggests research before generation works well")

        return tips

    async def get_task_patterns(
        self,
        hours: int = 168,  # 1 week
    ) -> Dict[str, TaskPattern]:
        """
        Get patterns for all task types.

        Args:
            hours: Time window in hours

        Returns:
            Dict mapping task_type to TaskPattern
        """
        # Check cache
        if (
            self._cache_updated_at and
            datetime.utcnow() - self._cache_updated_at < timedelta(minutes=self._cache_ttl_minutes) and
            self._pattern_cache
        ):
            return self._pattern_cache

        if not self.collector.db:
            return {}

        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Get all task types with stats
        result = await self.collector.db.execute(
            select(
                AgentTrajectory.task_type,
                func.count(AgentTrajectory.id).label("total"),
                func.sum(case((AgentTrajectory.success == True, 1), else_=0)).label("successes"),
            )
            .where(AgentTrajectory.created_at >= cutoff)
            .group_by(AgentTrajectory.task_type)
        )

        patterns = {}
        for row in result.all():
            task_type = row[0]
            total = row[1] or 0
            successes = row[2] or 0

            if total > 0:
                # Get common steps for this task type
                trajectories = await self._get_similar_trajectories(
                    task_type=task_type,
                    limit=20,
                )
                common_steps = await self._extract_common_steps(trajectories)

                patterns[task_type] = TaskPattern(
                    task_type=task_type,
                    common_steps=common_steps,
                    avg_step_count=sum(
                        len(t.trajectory_steps or []) for t in trajectories
                    ) / len(trajectories) if trajectories else 0,
                    success_rate=successes / total,
                    sample_count=total,
                )

        self._pattern_cache = patterns
        self._cache_updated_at = datetime.utcnow()

        return patterns


def get_adaptive_planner(collector: TrajectoryCollector) -> AdaptivePlanner:
    """Create an adaptive planner instance."""
    return AdaptivePlanner(collector)
