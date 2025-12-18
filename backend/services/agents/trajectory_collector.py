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
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import select, func, and_, desc
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
        stats_result = await self.db.execute(
            select(
                func.count(AgentTrajectory.id).label("total"),
                func.sum(
                    func.cast(AgentTrajectory.success, type_=func.INTEGER)
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
                    func.cast(AgentTrajectory.success, type_=func.INTEGER)
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
