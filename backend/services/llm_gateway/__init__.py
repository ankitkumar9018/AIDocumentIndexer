"""
AIDocumentIndexer - LLM Gateway
================================

Centralized LLM management with cost control and budget enforcement.

Features:
- OpenAI-compatible API proxy
- Budget enforcement (soft/hard limits)
- Virtual API keys for access control
- Usage tracking and analytics
- Multi-provider routing
"""

from backend.services.llm_gateway.gateway import LLMGateway, GatewayRequest, GatewayResponse
from backend.services.llm_gateway.budget import BudgetManager, Budget, BudgetPeriod
from backend.services.llm_gateway.virtual_keys import VirtualKeyManager, VirtualApiKey
from backend.services.llm_gateway.usage import UsageTracker, UsageRecord

__all__ = [
    # Gateway
    "LLMGateway",
    "GatewayRequest",
    "GatewayResponse",
    # Budget
    "BudgetManager",
    "Budget",
    "BudgetPeriod",
    # Virtual Keys
    "VirtualKeyManager",
    "VirtualApiKey",
    # Usage
    "UsageTracker",
    "UsageRecord",
]
