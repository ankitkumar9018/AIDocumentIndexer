"""
AIDocumentIndexer - Prompt Optimization Module
===============================================

Self-improvement system for agent prompts.

Components:
- PromptBuilderAgent: Analyzes failures and generates improved prompts
- PromptVersionManager: Manages versioning and A/B testing
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

__all__ = [
    "PromptBuilderAgent",
    "FailureAnalysis",
    "PromptMutation",
    "PromptVersionManager",
    "ABTestResult",
    "VariantResult",
]
