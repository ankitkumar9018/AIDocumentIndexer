"""
AIDocumentIndexer - Default Agent Seeding
==========================================

Seeds the database with default agent definitions on startup.
This ensures the multi-agent system has the required agents configured.
"""

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import AgentDefinition
from backend.db.database import async_session_context

logger = structlog.get_logger(__name__)

# Default agent definitions
DEFAULT_AGENTS = [
    {
        "name": "Manager Agent",
        "agent_type": "manager",
        "description": "Orchestrates task execution by analyzing requests, creating execution plans, and delegating to specialized worker agents.",
        "default_temperature": 0.3,
        "max_tokens": 2048,
        "settings": {
            "max_iterations": 10,
            "planning_enabled": True,
            "cost_estimation_enabled": True,
        },
        "is_active": True,
    },
    {
        "name": "Generator Agent",
        "agent_type": "generator",
        "description": "Generates content including drafts, summaries, reports, and creative writing based on retrieved context and user requirements.",
        "default_temperature": 0.7,
        "max_tokens": 4096,
        "settings": {
            "tools": ["generate", "summarize", "rewrite"],
            "supports_streaming": True,
        },
        "is_active": True,
    },
    {
        "name": "Critic Agent",
        "agent_type": "critic",
        "description": "Reviews and critiques generated content, checking for accuracy, completeness, tone, and alignment with requirements.",
        "default_temperature": 0.2,
        "max_tokens": 2048,
        "settings": {
            "tools": ["review", "fact_check", "improve"],
            "critique_aspects": ["accuracy", "completeness", "tone", "clarity"],
        },
        "is_active": True,
    },
    {
        "name": "Research Agent",
        "agent_type": "research",
        "description": "Searches and retrieves relevant information from the document corpus using RAG to gather context for other agents.",
        "default_temperature": 0.1,
        "max_tokens": 2048,
        "settings": {
            "tools": ["search", "retrieve", "compare"],
            "max_search_results": 10,
            "reranking_enabled": True,
        },
        "is_active": True,
    },
    {
        "name": "Tool Execution Agent",
        "agent_type": "tool_executor",
        "description": "Executes specialized tools and external integrations including web scraping, API calls, and data transformations.",
        "default_temperature": 0.0,
        "max_tokens": 1024,
        "settings": {
            "tools": ["web_scrape", "api_call", "transform"],
            "timeout_seconds": 30,
            "retry_attempts": 3,
        },
        "is_active": True,
    },
]


async def seed_default_agents() -> None:
    """
    Seed the database with default agent definitions if they don't exist.

    This function is idempotent - it only creates agents that don't already exist
    based on the agent_type field.
    """
    try:
        async with async_session_context() as db:
            # Check which agents already exist
            existing_result = await db.execute(
                select(AgentDefinition.agent_type)
            )
            existing_types = {row[0] for row in existing_result.fetchall()}

            # Create agents that don't exist
            created_count = 0
            for agent_data in DEFAULT_AGENTS:
                if agent_data["agent_type"] not in existing_types:
                    agent = AgentDefinition(**agent_data)
                    db.add(agent)
                    created_count += 1
                    logger.info(
                        "Creating default agent",
                        agent_type=agent_data["agent_type"],
                        name=agent_data["name"],
                    )

            if created_count > 0:
                await db.commit()
                logger.info(
                    "Default agents seeded successfully",
                    created=created_count,
                    total=len(DEFAULT_AGENTS),
                )
            else:
                logger.debug(
                    "All default agents already exist",
                    total=len(DEFAULT_AGENTS),
                )

    except Exception as e:
        logger.error("Failed to seed default agents", error=str(e))
        raise


async def get_active_agent_configs() -> dict:
    """
    Get configurations for all active agents from the database.

    Returns:
        dict: Mapping of agent_type to agent configuration
    """
    async with async_session_context() as db:
        result = await db.execute(
            select(AgentDefinition).where(AgentDefinition.is_active == True)
        )
        agents = result.scalars().all()

        return {
            agent.agent_type: {
                "id": str(agent.id),
                "name": agent.name,
                "description": agent.description,
                "temperature": agent.default_temperature,
                "max_tokens": agent.max_tokens,
                "settings": agent.settings or {},
                "provider_id": str(agent.default_provider_id) if agent.default_provider_id else None,
                "model": agent.default_model,
            }
            for agent in agents
        }
