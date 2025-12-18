"""
AIDocumentIndexer - Prompt Version Manager
==========================================

Manages prompt versions with A/B testing capabilities.

Features:
- Version history with lineage tracking
- A/B testing with traffic splitting
- Rainbow deployment (gradual rollout)
- Human approval workflow
- Rollback capability
"""

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import (
    AgentDefinition,
    AgentPromptVersion,
    AgentTrajectory,
    PromptOptimizationJob,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VariantResult:
    """Result statistics for a prompt variant."""
    variant_id: str
    version_number: int
    execution_count: int
    success_count: int
    success_rate: float
    avg_quality_score: Optional[float]
    traffic_percentage: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "version_number": self.version_number,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "avg_quality_score": self.avg_quality_score,
            "traffic_percentage": self.traffic_percentage,
        }


@dataclass
class ABTestResult:
    """Results of an A/B test."""
    test_id: str
    agent_id: str
    status: str  # running, completed, cancelled
    started_at: datetime
    ended_at: Optional[datetime]
    baseline_variant: VariantResult
    test_variants: List[VariantResult] = field(default_factory=list)
    winning_variant_id: Optional[str] = None
    improvement_percentage: Optional[float] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "agent_id": self.agent_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "baseline_variant": self.baseline_variant.to_dict(),
            "test_variants": [v.to_dict() for v in self.test_variants],
            "winning_variant_id": self.winning_variant_id,
            "improvement_percentage": self.improvement_percentage,
            "confidence": self.confidence,
        }


# =============================================================================
# Prompt Version Manager
# =============================================================================

class PromptVersionManager:
    """
    Manages prompt versions with A/B testing support.

    Supports:
    - Traffic splitting for A/B tests
    - Rainbow deployment (10% → 25% → 50% → 100%)
    - Human approval before promotion
    - Rollback to previous versions
    """

    # Rainbow deployment stages
    RAINBOW_STAGES = [10, 25, 50, 100]  # Traffic percentages

    def __init__(self, db: AsyncSession):
        """
        Initialize version manager.

        Args:
            db: Database session
        """
        self.db = db

    async def get_prompt_for_execution(
        self,
        agent_id: str,
    ) -> Optional[AgentPromptVersion]:
        """
        Get prompt version for execution, respecting A/B test traffic splits.

        Args:
            agent_id: Agent UUID

        Returns:
            Selected AgentPromptVersion
        """
        agent_uuid = uuid.UUID(agent_id)

        # Get all active versions
        result = await self.db.execute(
            select(AgentPromptVersion)
            .where(and_(
                AgentPromptVersion.agent_id == agent_uuid,
                AgentPromptVersion.is_active == True,
            ))
        )
        active_versions = list(result.scalars().all())

        if not active_versions:
            return None

        if len(active_versions) == 1:
            return active_versions[0]

        # Weighted random selection based on traffic_percentage
        total_traffic = sum(v.traffic_percentage for v in active_versions)
        if total_traffic <= 0:
            return active_versions[0]

        rand = random.randint(1, total_traffic)
        cumulative = 0

        for version in active_versions:
            cumulative += version.traffic_percentage
            if rand <= cumulative:
                return version

        return active_versions[0]

    async def get_version(
        self,
        version_id: str,
    ) -> Optional[AgentPromptVersion]:
        """Get a specific prompt version."""
        result = await self.db.execute(
            select(AgentPromptVersion)
            .where(AgentPromptVersion.id == uuid.UUID(version_id))
        )
        return result.scalar_one_or_none()

    async def get_version_history(
        self,
        agent_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get prompt version history with performance metrics.

        Args:
            agent_id: Agent UUID
            limit: Maximum results

        Returns:
            List of version info dicts
        """
        agent_uuid = uuid.UUID(agent_id)

        result = await self.db.execute(
            select(AgentPromptVersion)
            .where(AgentPromptVersion.agent_id == agent_uuid)
            .order_by(AgentPromptVersion.version_number.desc())
            .limit(limit)
        )
        versions = result.scalars().all()

        history = []
        for v in versions:
            history.append({
                "id": str(v.id),
                "version_number": v.version_number,
                "change_reason": v.change_reason,
                "created_by": v.created_by,
                "created_at": v.created_at.isoformat() if v.created_at else None,
                "is_active": v.is_active,
                "traffic_percentage": v.traffic_percentage,
                "execution_count": v.execution_count,
                "success_count": v.success_count,
                "success_rate": (
                    v.success_count / v.execution_count
                    if v.execution_count > 0 else None
                ),
                "avg_quality_score": v.avg_quality_score,
            })

        return history

    async def get_active_versions(
        self,
        agent_id: str,
    ) -> List[AgentPromptVersion]:
        """Get all active versions for an agent."""
        result = await self.db.execute(
            select(AgentPromptVersion)
            .where(and_(
                AgentPromptVersion.agent_id == uuid.UUID(agent_id),
                AgentPromptVersion.is_active == True,
            ))
        )
        return list(result.scalars().all())

    # =========================================================================
    # A/B Testing
    # =========================================================================

    async def start_ab_test(
        self,
        agent_id: str,
        baseline_version_id: str,
        test_version_ids: List[str],
        initial_traffic_percentage: int = 10,
    ) -> Dict[str, Any]:
        """
        Start an A/B test with traffic splitting.

        Args:
            agent_id: Agent UUID
            baseline_version_id: Current production version
            test_version_ids: Versions to test
            initial_traffic_percentage: Initial traffic % per test variant

        Returns:
            Test configuration
        """
        # Activate test versions with initial traffic
        total_test_traffic = len(test_version_ids) * initial_traffic_percentage
        baseline_traffic = max(100 - total_test_traffic, 50)  # Baseline keeps majority

        # Update baseline
        baseline = await self.get_version(baseline_version_id)
        if baseline:
            baseline.is_active = True
            baseline.traffic_percentage = baseline_traffic

        # Activate test versions
        for version_id in test_version_ids:
            version = await self.get_version(version_id)
            if version:
                version.is_active = True
                version.traffic_percentage = initial_traffic_percentage

        await self.db.commit()

        logger.info(
            "Started A/B test",
            agent_id=agent_id,
            baseline_id=baseline_version_id,
            test_versions=test_version_ids,
            baseline_traffic=baseline_traffic,
            test_traffic=initial_traffic_percentage,
        )

        return {
            "status": "running",
            "baseline_version_id": baseline_version_id,
            "baseline_traffic": baseline_traffic,
            "test_versions": [
                {"version_id": vid, "traffic": initial_traffic_percentage}
                for vid in test_version_ids
            ],
        }

    async def advance_rainbow_stage(
        self,
        agent_id: str,
        version_id: str,
        current_traffic: int,
    ) -> int:
        """
        Advance a version to the next rainbow deployment stage.

        Args:
            agent_id: Agent UUID
            version_id: Version to advance
            current_traffic: Current traffic percentage

        Returns:
            New traffic percentage
        """
        # Find next stage
        next_traffic = current_traffic
        for stage in self.RAINBOW_STAGES:
            if stage > current_traffic:
                next_traffic = stage
                break

        if next_traffic == current_traffic:
            return current_traffic  # Already at 100%

        # Update version
        version = await self.get_version(version_id)
        if version:
            version.traffic_percentage = next_traffic

            # Reduce other active versions proportionally
            other_versions = await self.get_active_versions(agent_id)
            remaining_traffic = 100 - next_traffic
            other_count = len([v for v in other_versions if str(v.id) != version_id])

            if other_count > 0:
                per_other = remaining_traffic // other_count
                for v in other_versions:
                    if str(v.id) != version_id:
                        v.traffic_percentage = per_other

            await self.db.commit()

        logger.info(
            "Advanced rainbow stage",
            version_id=version_id,
            old_traffic=current_traffic,
            new_traffic=next_traffic,
        )

        return next_traffic

    async def get_test_results(
        self,
        agent_id: str,
        hours: int = 24,
    ) -> ABTestResult:
        """
        Get current A/B test results.

        Args:
            agent_id: Agent UUID
            hours: Analysis window

        Returns:
            ABTestResult with variant statistics
        """
        active_versions = await self.get_active_versions(agent_id)

        if not active_versions:
            raise ValueError(f"No active versions for agent {agent_id}")

        # Get stats for each version
        variant_results = []
        baseline = None

        for version in active_versions:
            stats = await self._get_version_stats(str(version.id), hours)

            result = VariantResult(
                variant_id=str(version.id),
                version_number=version.version_number,
                execution_count=stats["execution_count"],
                success_count=stats["success_count"],
                success_rate=stats["success_rate"],
                avg_quality_score=stats["avg_quality_score"],
                traffic_percentage=version.traffic_percentage,
            )

            # Assume highest traffic percentage is baseline
            if baseline is None or version.traffic_percentage > baseline.traffic_percentage:
                if baseline:
                    variant_results.append(baseline)
                baseline = result
            else:
                variant_results.append(result)

        # Determine winner
        all_variants = [baseline] + variant_results
        winner = max(
            all_variants,
            key=lambda v: (v.success_rate * 0.7 + (v.avg_quality_score or 3) / 5 * 0.3)
        )

        # Calculate improvement
        improvement = None
        if baseline and winner.variant_id != baseline.variant_id:
            if baseline.success_rate > 0:
                improvement = (
                    (winner.success_rate - baseline.success_rate) / baseline.success_rate * 100
                )

        return ABTestResult(
            test_id=str(uuid.uuid4()),
            agent_id=agent_id,
            status="running",
            started_at=datetime.utcnow() - timedelta(hours=hours),
            ended_at=None,
            baseline_variant=baseline,
            test_variants=variant_results,
            winning_variant_id=winner.variant_id,
            improvement_percentage=improvement,
            confidence=self._calculate_confidence(all_variants),
        )

    async def _get_version_stats(
        self,
        version_id: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """Get statistics for a prompt version."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        version_uuid = uuid.UUID(version_id)

        result = await self.db.execute(
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
        row = result.one()

        total = row.total or 0
        successes = row.successes or 0

        return {
            "execution_count": total,
            "success_count": successes,
            "success_rate": (successes / total) if total > 0 else 0.0,
            "avg_quality_score": float(row.avg_quality) if row.avg_quality else None,
        }

    def _calculate_confidence(
        self,
        variants: List[VariantResult],
    ) -> float:
        """Calculate statistical confidence in test results."""
        # Simple confidence based on sample size
        total_executions = sum(v.execution_count for v in variants)

        if total_executions < 20:
            return 0.3
        elif total_executions < 50:
            return 0.5
        elif total_executions < 100:
            return 0.7
        elif total_executions < 200:
            return 0.85
        else:
            return 0.95

    # =========================================================================
    # Promotion and Rollback
    # =========================================================================

    async def promote_version(
        self,
        agent_id: str,
        version_id: str,
        approver_id: str,
    ) -> Dict[str, Any]:
        """
        Promote a version to 100% traffic (production).

        Args:
            agent_id: Agent UUID
            version_id: Version to promote
            approver_id: User who approved

        Returns:
            Promotion result
        """
        agent_uuid = uuid.UUID(agent_id)

        # Deactivate other versions
        await self._deactivate_other_versions(agent_id, version_id)

        # Activate winning version at 100%
        version = await self.get_version(version_id)
        if version:
            version.is_active = True
            version.traffic_percentage = 100

        # Update agent's active prompt
        result = await self.db.execute(
            select(AgentDefinition)
            .where(AgentDefinition.id == agent_uuid)
        )
        agent = result.scalar_one_or_none()
        if agent:
            agent.active_prompt_version_id = uuid.UUID(version_id)

        await self.db.commit()

        logger.info(
            "Promoted prompt version",
            agent_id=agent_id,
            version_id=version_id,
            approver_id=approver_id,
        )

        return {
            "success": True,
            "promoted_version_id": version_id,
            "approver_id": approver_id,
        }

    async def rollback(
        self,
        agent_id: str,
        target_version_id: str,
    ) -> Dict[str, Any]:
        """
        Rollback to a previous prompt version.

        Args:
            agent_id: Agent UUID
            target_version_id: Version to rollback to

        Returns:
            Rollback result
        """
        agent_uuid = uuid.UUID(agent_id)

        # Verify target version exists and belongs to this agent
        target = await self.get_version(target_version_id)
        if not target or str(target.agent_id) != agent_id:
            return {
                "success": False,
                "error": "Target version not found or doesn't belong to agent",
            }

        # Deactivate all other versions
        await self._deactivate_other_versions(agent_id, target_version_id)

        # Activate target version
        target.is_active = True
        target.traffic_percentage = 100

        # Update agent
        result = await self.db.execute(
            select(AgentDefinition)
            .where(AgentDefinition.id == agent_uuid)
        )
        agent = result.scalar_one_or_none()
        if agent:
            agent.active_prompt_version_id = uuid.UUID(target_version_id)

        await self.db.commit()

        logger.info(
            "Rolled back to version",
            agent_id=agent_id,
            version_id=target_version_id,
            version_number=target.version_number,
        )

        return {
            "success": True,
            "rolled_back_to": target_version_id,
            "version_number": target.version_number,
        }

    async def _deactivate_other_versions(
        self,
        agent_id: str,
        keep_version_id: str,
    ) -> None:
        """Deactivate all versions except the specified one."""
        result = await self.db.execute(
            select(AgentPromptVersion)
            .where(and_(
                AgentPromptVersion.agent_id == uuid.UUID(agent_id),
                AgentPromptVersion.id != uuid.UUID(keep_version_id),
            ))
        )
        versions = result.scalars().all()

        for v in versions:
            v.is_active = False
            v.traffic_percentage = 0

    # =========================================================================
    # Version Creation
    # =========================================================================

    async def create_version(
        self,
        agent_id: str,
        system_prompt: str,
        task_prompt_template: str,
        change_reason: str,
        created_by: str = "manual",
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> AgentPromptVersion:
        """
        Create a new prompt version.

        Args:
            agent_id: Agent UUID
            system_prompt: System prompt text
            task_prompt_template: Task template with {{placeholders}}
            change_reason: Why this version was created
            created_by: Creator (manual, prompt_builder, system)
            few_shot_examples: Optional examples
            output_schema: Optional output format schema

        Returns:
            Created version
        """
        agent_uuid = uuid.UUID(agent_id)

        # Get max version number
        result = await self.db.execute(
            select(func.max(AgentPromptVersion.version_number))
            .where(AgentPromptVersion.agent_id == agent_uuid)
        )
        max_version = result.scalar() or 0

        # Get current active version as parent
        result = await self.db.execute(
            select(AgentPromptVersion.id)
            .where(and_(
                AgentPromptVersion.agent_id == agent_uuid,
                AgentPromptVersion.is_active == True,
            ))
            .limit(1)
        )
        parent_id = result.scalar()

        version = AgentPromptVersion(
            agent_id=agent_uuid,
            version_number=max_version + 1,
            system_prompt=system_prompt,
            task_prompt_template=task_prompt_template,
            few_shot_examples=few_shot_examples or [],
            output_schema=output_schema,
            change_reason=change_reason,
            created_by=created_by,
            is_active=False,
            traffic_percentage=0,
            parent_version_id=parent_id,
        )

        self.db.add(version)
        await self.db.commit()
        await self.db.refresh(version)

        logger.info(
            "Created prompt version",
            version_id=str(version.id),
            version_number=version.version_number,
            agent_id=agent_id,
        )

        return version

    async def update_version_stats(
        self,
        version_id: str,
        success: bool,
        quality_score: Optional[float] = None,
    ) -> None:
        """
        Update version execution statistics.

        Called after each execution using this version.

        Args:
            version_id: Version UUID
            success: Whether execution succeeded
            quality_score: Optional quality score
        """
        version = await self.get_version(version_id)
        if not version:
            return

        version.execution_count = (version.execution_count or 0) + 1
        if success:
            version.success_count = (version.success_count or 0) + 1

        if quality_score is not None:
            # Running average
            if version.avg_quality_score is not None:
                total_scores = version.avg_quality_score * (version.execution_count - 1)
                version.avg_quality_score = (total_scores + quality_score) / version.execution_count
            else:
                version.avg_quality_score = quality_score

        await self.db.commit()
