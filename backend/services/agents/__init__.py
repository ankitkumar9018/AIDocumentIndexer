"""
AIDocumentIndexer - Multi-Agent System
=======================================

Production-grade multi-agent orchestration with self-improvement capabilities.

Components:
- BaseAgent: Abstract base class for all agents
- AgentTask: Task definitions with validation
- AgentResult: Standardized result format
- TrajectoryCollector: Execution recording for analysis
- ManagerAgent: Orchestrator agent
- Worker Agents: Generator, Critic, Research, ToolExecution
"""

from backend.services.agents.agent_base import (
    AgentTask,
    AgentResult,
    ValidationResult,
    BaseAgent,
    AgentConfig,
    FallbackStrategy,
    TaskStatus,
)

__all__ = [
    "AgentTask",
    "AgentResult",
    "ValidationResult",
    "BaseAgent",
    "AgentConfig",
    "FallbackStrategy",
    "TaskStatus",
]
