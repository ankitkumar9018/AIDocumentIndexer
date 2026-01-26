"""
AIDocumentIndexer - Prompt Optimization Module
===============================================

Self-improvement system for agent prompts.

Components:
- PromptBuilderAgent: Analyzes failures and generates improved prompts
- PromptVersionManager: Manages versioning and A/B testing
- DSPyOptimizer: Automated prompt optimization via DSPy compilation (Phase 93)
- DSPyExampleCollector: Training data collection from user interactions
"""

from backend.services.prompt_optimization.prompt_builder_agent import (
    PromptBuilderAgent,
    FailureAnalysis,
    PromptMutation,
)
from backend.services.prompt_optimization.prompt_version_manager import (
    PromptVersionManager,
    ABTestResult,
    VariantResult,
)
from backend.services.prompt_optimization.dspy_optimizer import (
    DSPyOptimizer,
    DSPyOptimizationResult,
)
from backend.services.prompt_optimization.dspy_example_collector import (
    DSPyExampleCollector,
)

__all__ = [
    "PromptBuilderAgent",
    "FailureAnalysis",
    "PromptMutation",
    "PromptVersionManager",
    "ABTestResult",
    "VariantResult",
    "DSPyOptimizer",
    "DSPyOptimizationResult",
    "DSPyExampleCollector",
]
