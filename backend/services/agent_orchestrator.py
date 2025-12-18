"""
AIDocumentIndexer - Agent Orchestrator Service
===============================================

Central orchestration service that coordinates:
1. Manager Agent for planning
2. Worker Agents for execution
3. Trajectory collection for analysis
4. Cost estimation and budget enforcement
5. Plan persistence for recovery

This is the main entry point for agent-mode execution.
"""

import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import (
    AgentDefinition,
    AgentExecutionPlan,
    AgentTrajectory,
    ExecutionModePreference,
)
from backend.services.agents.agent_base import (
    AgentConfig,
    AgentTask,
    TaskType,
    TaskStatus,
)
from backend.services.agents.manager_agent import (
    ManagerAgent,
    ExecutionPlan,
    PlanStatus,
)
from backend.services.agents.worker_agents import create_worker_agents
from backend.services.agents.trajectory_collector import TrajectoryCollector

logger = structlog.get_logger(__name__)


class AgentOrchestrator:
    """
    Main orchestration service for multi-agent execution.

    Coordinates:
    - Request planning via Manager Agent
    - Execution via Worker Agents
    - Trajectory collection
    - Cost estimation and budget checks
    - Plan persistence
    """

    def __init__(
        self,
        db: AsyncSession,
        rag_service=None,
        scraper_service=None,
        generator_service=None,
        cost_estimator=None,
    ):
        """
        Initialize orchestrator.

        Args:
            db: Database session
            rag_service: RAGService for research
            scraper_service: ScraperService for web research
            generator_service: GeneratorService for document generation
            cost_estimator: AgentCostEstimator for budget management
        """
        self.db = db
        self.rag_service = rag_service
        self.scraper_service = scraper_service
        self.generator_service = generator_service
        self.cost_estimator = cost_estimator

        # Initialize trajectory collector
        self.trajectory_collector = TrajectoryCollector(db)

        # Initialize worker agents
        self.workers = create_worker_agents(
            rag_service=rag_service,
            scraper_service=scraper_service,
            generator_service=generator_service,
            trajectory_collector=self.trajectory_collector,
        )

        # Initialize manager agent
        manager_config = AgentConfig(
            agent_id=str(uuid.uuid4()),
            name="Manager Agent",
            description="Task decomposition and orchestration",
        )
        self.manager = ManagerAgent(
            config=manager_config,
            trajectory_collector=self.trajectory_collector,
            worker_registry=self.workers,
        )

        logger.info(
            "Agent orchestrator initialized",
            worker_count=len(self.workers),
        )

    def set_services(
        self,
        rag_service=None,
        scraper_service=None,
        generator_service=None,
        cost_estimator=None,
    ) -> None:
        """Update service instances."""
        if rag_service:
            self.rag_service = rag_service
            if "research" in self.workers:
                self.workers["research"].set_services(rag_service=rag_service)
        if scraper_service:
            self.scraper_service = scraper_service
            if "research" in self.workers:
                self.workers["research"].set_services(scraper_service=scraper_service)
        if generator_service:
            self.generator_service = generator_service
            if "tool" in self.workers:
                self.workers["tool"].set_services(generator_service=generator_service)
        if cost_estimator:
            self.cost_estimator = cost_estimator

    async def process_request(
        self,
        request: str,
        session_id: str,
        user_id: str,
        show_cost_estimation: bool = True,
        require_approval_above_usd: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a request through the multi-agent system.

        Args:
            request: User request text
            session_id: Session UUID
            user_id: User UUID
            show_cost_estimation: Whether to show cost before execution
            require_approval_above_usd: Threshold for requiring approval
            context: Additional context

        Yields:
            Progress updates and final result
        """
        context = context or {}

        # Yield start notification
        yield {
            "type": "processing_started",
            "session_id": session_id,
            "request": request,
        }

        try:
            # 1. Create execution plan
            yield {"type": "planning", "message": "Creating execution plan..."}

            plan = await self.manager.plan_execution(
                user_request=request,
                session_id=session_id,
                user_id=user_id,
                context=context,
            )

            yield {
                "type": "plan_created",
                "plan_id": plan.id,
                "summary": plan.summary(),
                "step_count": len(plan.steps),
            }

            # 2. Cost estimation
            if self.cost_estimator and show_cost_estimation:
                yield {"type": "estimating_cost", "message": "Estimating execution cost..."}

                cost_estimate = await self.cost_estimator.estimate_plan_cost(plan)
                plan.total_estimated_cost_usd = cost_estimate.total_cost_usd

                yield {
                    "type": "cost_estimated",
                    "estimated_cost_usd": cost_estimate.total_cost_usd,
                    "step_costs": [s.to_dict() for s in cost_estimate.steps] if hasattr(cost_estimate, 'steps') else [],
                }

                # 3. Budget check
                budget_check = await self.cost_estimator.check_budget(
                    user_id, cost_estimate.total_cost_usd
                )

                if not budget_check.allowed:
                    yield {
                        "type": "budget_exceeded",
                        "estimated_cost": cost_estimate.total_cost_usd,
                        "remaining_budget": budget_check.remaining_budget,
                        "message": budget_check.reason,
                    }
                    # Save plan as cancelled
                    await self._persist_plan(plan, PlanStatus.CANCELLED)
                    return

                # 4. Check if approval needed
                if cost_estimate.total_cost_usd > require_approval_above_usd:
                    plan.status = PlanStatus.AWAITING_APPROVAL

                    yield {
                        "type": "approval_required",
                        "plan_id": plan.id,
                        "estimated_cost_usd": cost_estimate.total_cost_usd,
                        "threshold_usd": require_approval_above_usd,
                        "message": "Cost exceeds threshold. Awaiting approval.",
                    }

                    # Persist plan for later approval
                    await self._persist_plan(plan, PlanStatus.AWAITING_APPROVAL)
                    return

            # 5. Persist plan
            await self._persist_plan(plan, PlanStatus.EXECUTING)

            # 6. Execute plan
            async for update in self.manager.execute_plan(plan, context):
                # Forward all updates
                yield update

                # Update persisted plan on completion
                if update.get("type") == "plan_completed":
                    await self._update_plan_status(
                        plan.id,
                        PlanStatus.COMPLETED if update.get("status") == "success" else PlanStatus.FAILED,
                        final_output=update.get("output"),
                        actual_cost=update.get("total_cost_usd"),
                    )

        except Exception as e:
            logger.error(f"Orchestration error: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
            }

    async def approve_execution(
        self,
        plan_id: str,
        user_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Approve and execute a plan that was awaiting approval.

        Args:
            plan_id: Plan UUID
            user_id: Approving user UUID

        Yields:
            Execution progress updates
        """
        # Load plan from database
        plan = await self._load_plan(plan_id)

        if not plan:
            yield {
                "type": "error",
                "error": f"Plan {plan_id} not found",
            }
            return

        if plan.status != PlanStatus.AWAITING_APPROVAL:
            yield {
                "type": "error",
                "error": f"Plan is not awaiting approval (status: {plan.status})",
            }
            return

        # Mark user approval
        plan.user_approved_cost = True
        await self._update_plan_status(plan_id, PlanStatus.EXECUTING)

        yield {
            "type": "approval_confirmed",
            "plan_id": plan_id,
        }

        # Rebuild ExecutionPlan from database record
        execution_plan = self._rebuild_execution_plan(plan)

        # Execute
        async for update in self.manager.execute_plan(execution_plan, {}):
            yield update

            if update.get("type") == "plan_completed":
                await self._update_plan_status(
                    plan_id,
                    PlanStatus.COMPLETED if update.get("status") == "success" else PlanStatus.FAILED,
                    final_output=update.get("output"),
                    actual_cost=update.get("total_cost_usd"),
                )

    async def cancel_execution(
        self,
        plan_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Cancel a pending or running execution.

        Args:
            plan_id: Plan UUID
            user_id: User UUID

        Returns:
            Cancellation result
        """
        plan = await self._load_plan(plan_id)

        if not plan:
            return {"success": False, "error": f"Plan {plan_id} not found"}

        if plan.status not in (PlanStatus.PENDING, PlanStatus.AWAITING_APPROVAL, PlanStatus.EXECUTING):
            return {"success": False, "error": f"Cannot cancel plan with status {plan.status}"}

        await self._update_plan_status(plan_id, PlanStatus.CANCELLED)

        return {
            "success": True,
            "plan_id": plan_id,
            "status": "cancelled",
        }

    async def get_plan_status(
        self,
        plan_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get status of an execution plan.

        Args:
            plan_id: Plan UUID

        Returns:
            Plan status dict or None
        """
        plan = await self._load_plan(plan_id)

        if not plan:
            return None

        return {
            "plan_id": str(plan.id),
            "status": plan.status,
            "user_request": plan.user_request,
            "step_count": len(plan.plan_steps) if plan.plan_steps else 0,
            "current_step": plan.current_step,
            "completed_steps": plan.completed_steps,
            "estimated_cost_usd": plan.estimated_cost_usd,
            "actual_cost_usd": plan.actual_cost_usd,
            "created_at": plan.created_at.isoformat() if plan.created_at else None,
            "started_at": plan.started_at.isoformat() if plan.started_at else None,
            "completed_at": plan.completed_at.isoformat() if plan.completed_at else None,
        }

    async def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of all agents.

        Returns:
            Dict with agent status information
        """
        agents = []

        # Manager
        agents.append({
            "id": self.manager.agent_id,
            "name": self.manager.name,
            "type": "manager",
            "model": self.manager.config.model,
            "status": "active",
        })

        # Workers
        for agent_type, agent in self.workers.items():
            agents.append({
                "id": agent.agent_id,
                "name": agent.name,
                "type": agent_type,
                "model": agent.config.model,
                "status": "active",
            })

        return {
            "agents": agents,
            "total_count": len(agents),
            "active_count": len(agents),
        }

    async def get_agent_metrics(
        self,
        agent_id: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get performance metrics for an agent.

        Args:
            agent_id: Agent UUID
            hours: Time window

        Returns:
            Metrics dict
        """
        return await self.trajectory_collector.get_agent_stats(agent_id, hours)

    # =========================================================================
    # Plan Persistence
    # =========================================================================

    async def _persist_plan(
        self,
        plan: ExecutionPlan,
        status: PlanStatus,
    ) -> AgentExecutionPlan:
        """Persist execution plan to database."""
        db_plan = AgentExecutionPlan(
            id=uuid.UUID(plan.id),
            session_id=uuid.UUID(plan.session_id),
            user_id=uuid.UUID(plan.user_id),
            user_request=plan.user_request,
            plan_steps=[s.to_dict() for s in plan.steps],
            status=status.value,
            current_step=plan.current_step_index,
            completed_steps=plan.completed_steps,
            estimated_cost_usd=plan.total_estimated_cost_usd,
            started_at=plan.started_at,
        )

        self.db.add(db_plan)
        await self.db.commit()
        await self.db.refresh(db_plan)

        logger.info(f"Persisted plan {plan.id} with status {status.value}")

        return db_plan

    async def _load_plan(
        self,
        plan_id: str,
    ) -> Optional[AgentExecutionPlan]:
        """Load plan from database."""
        result = await self.db.execute(
            select(AgentExecutionPlan)
            .where(AgentExecutionPlan.id == uuid.UUID(plan_id))
        )
        return result.scalar_one_or_none()

    async def _update_plan_status(
        self,
        plan_id: str,
        status: PlanStatus,
        final_output: Optional[str] = None,
        actual_cost: Optional[float] = None,
    ) -> None:
        """Update plan status in database."""
        plan = await self._load_plan(plan_id)
        if plan:
            plan.status = status.value
            if final_output:
                plan.final_output = final_output
            if actual_cost:
                plan.actual_cost_usd = actual_cost
            if status in (PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED):
                plan.completed_at = datetime.utcnow()

            await self.db.commit()

    def _rebuild_execution_plan(
        self,
        db_plan: AgentExecutionPlan,
    ) -> ExecutionPlan:
        """Rebuild ExecutionPlan from database record."""
        from backend.services.agents.manager_agent import PlanStep

        steps = []
        for step_data in db_plan.plan_steps or []:
            task = AgentTask(
                id=step_data.get("task_id", str(uuid.uuid4())),
                type=TaskType(step_data.get("task_type", "generation")),
                name=step_data.get("task_name", "Task"),
                description=step_data.get("description", ""),
            )
            steps.append(PlanStep(
                id=step_data.get("id", str(uuid.uuid4())),
                agent_type=step_data.get("agent_type", "generator"),
                task=task,
                status=TaskStatus(step_data.get("status", "pending")),
                dependencies=step_data.get("dependencies", []),
            ))

        return ExecutionPlan(
            id=str(db_plan.id),
            session_id=str(db_plan.session_id),
            user_id=str(db_plan.user_id),
            user_request=db_plan.user_request,
            request_type="other",
            steps=steps,
            status=PlanStatus(db_plan.status),
            current_step_index=db_plan.current_step,
            total_estimated_cost_usd=db_plan.estimated_cost_usd or 0.0,
            created_at=db_plan.created_at,
            started_at=db_plan.started_at,
        )

    # =========================================================================
    # History and Analytics
    # =========================================================================

    async def get_execution_history(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get user's execution history.

        Args:
            user_id: User UUID
            limit: Maximum results

        Returns:
            List of plan summaries
        """
        result = await self.db.execute(
            select(AgentExecutionPlan)
            .where(AgentExecutionPlan.user_id == uuid.UUID(user_id))
            .order_by(AgentExecutionPlan.created_at.desc())
            .limit(limit)
        )
        plans = result.scalars().all()

        return [
            {
                "plan_id": str(p.id),
                "request": p.user_request[:100] + "..." if len(p.user_request) > 100 else p.user_request,
                "status": p.status,
                "step_count": len(p.plan_steps) if p.plan_steps else 0,
                "actual_cost_usd": p.actual_cost_usd,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in plans
        ]

    async def get_trajectories_for_plan(
        self,
        plan_id: str,
    ) -> List[AgentTrajectory]:
        """
        Get all trajectories associated with a plan.

        Args:
            plan_id: Plan UUID

        Returns:
            List of trajectories
        """
        # Load plan to get session_id
        plan = await self._load_plan(plan_id)
        if not plan:
            return []

        return await self.trajectory_collector.get_session_trajectories(
            str(plan.session_id)
        )


# Factory function for dependency injection
async def create_orchestrator(
    db: AsyncSession,
    rag_service=None,
    scraper_service=None,
    generator_service=None,
) -> AgentOrchestrator:
    """
    Create an AgentOrchestrator instance.

    Args:
        db: Database session
        rag_service: RAGService instance
        scraper_service: ScraperService instance
        generator_service: GeneratorService instance

    Returns:
        Configured AgentOrchestrator
    """
    return AgentOrchestrator(
        db=db,
        rag_service=rag_service,
        scraper_service=scraper_service,
        generator_service=generator_service,
    )
