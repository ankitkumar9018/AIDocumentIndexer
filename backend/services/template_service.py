"""
AIDocumentIndexer - Template Service
=====================================

Service for managing agent and workflow templates.

Provides:
- Loading templates from JSON files
- Template validation
- Template customization
- Creating agents/workflows from templates
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

# Template directory paths
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
AGENTS_TEMPLATES_DIR = TEMPLATES_DIR / "agents"
WORKFLOWS_TEMPLATES_DIR = TEMPLATES_DIR / "workflows"


class TemplateType(str, Enum):
    """Types of templates available."""
    AGENT = "agent"
    WORKFLOW = "workflow"


class AgentType(str, Enum):
    """Types of AI agents."""
    CHATBOT = "chatbot"
    VOICE = "voice"
    API_ONLY = "api_only"


@dataclass
class TemplateInfo:
    """Basic template information."""
    id: str
    name: str
    description: str
    type: TemplateType
    version: str
    category: Optional[str] = None
    complexity: Optional[str] = None
    patterns_used: Optional[List[str]] = None


@dataclass
class AgentTemplate:
    """Full agent template with all configurations."""
    id: str
    name: str
    description: str
    type: AgentType
    version: str
    system_prompt: str
    capabilities: Dict[str, bool]
    enhancements: Dict[str, Any]
    llm_config: Dict[str, Any]
    embedding_config: Optional[Dict[str, Any]] = None
    voice_config: Optional[Dict[str, Any]] = None
    ui_config: Optional[Dict[str, Any]] = None
    rate_limits: Optional[Dict[str, int]] = None
    languages: List[str] = None
    raw_config: Dict[str, Any] = None


@dataclass
class WorkflowTemplate:
    """Full workflow template with all nodes and configurations."""
    id: str
    name: str
    description: str
    version: str
    trigger: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    variables: Dict[str, Any]
    config: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    error_handling: Optional[Dict[str, Any]] = None
    observability: Optional[Dict[str, Any]] = None
    security: Optional[Dict[str, Any]] = None
    raw_config: Dict[str, Any] = None


class TemplateService:
    """Service for managing agent and workflow templates."""

    def __init__(self):
        self._agent_templates: Dict[str, AgentTemplate] = {}
        self._workflow_templates: Dict[str, WorkflowTemplate] = {}
        self._loaded = False

    async def initialize(self) -> None:
        """Load all templates from disk."""
        if self._loaded:
            return

        await self._load_agent_templates()
        await self._load_workflow_templates()
        self._loaded = True

        logger.info(
            "Template service initialized",
            agent_templates=len(self._agent_templates),
            workflow_templates=len(self._workflow_templates),
        )

    async def _load_agent_templates(self) -> None:
        """Load agent templates from the agents directory."""
        if not AGENTS_TEMPLATES_DIR.exists():
            logger.warning("Agents template directory not found", path=str(AGENTS_TEMPLATES_DIR))
            return

        for file_path in AGENTS_TEMPLATES_DIR.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    config = json.load(f)

                template_id = file_path.stem
                template = AgentTemplate(
                    id=template_id,
                    name=config.get("name", template_id),
                    description=config.get("description", ""),
                    type=AgentType(config.get("type", "chatbot").lower()),
                    version=config.get("version", "1.0.0"),
                    system_prompt=config.get("system_prompt", ""),
                    capabilities=config.get("capabilities", {}),
                    enhancements=config.get("enhancements", {}),
                    llm_config=config.get("llm_config", {}),
                    embedding_config=config.get("embedding_config"),
                    voice_config=config.get("voice_config"),
                    ui_config=config.get("ui_config"),
                    rate_limits=config.get("rate_limits"),
                    languages=config.get("languages", ["en"]),
                    raw_config=config,
                )
                self._agent_templates[template_id] = template
                logger.debug("Loaded agent template", template_id=template_id)

            except Exception as e:
                logger.error("Failed to load agent template", file=str(file_path), error=str(e))

    async def _load_workflow_templates(self) -> None:
        """Load workflow templates from the workflows directory."""
        if not WORKFLOWS_TEMPLATES_DIR.exists():
            logger.warning("Workflows template directory not found", path=str(WORKFLOWS_TEMPLATES_DIR))
            return

        for file_path in WORKFLOWS_TEMPLATES_DIR.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    config = json.load(f)

                template_id = file_path.stem
                template = WorkflowTemplate(
                    id=template_id,
                    name=config.get("name", template_id),
                    description=config.get("description", ""),
                    version=config.get("version", "1.0.0"),
                    trigger=config.get("trigger", "MANUAL"),
                    nodes=config.get("nodes", []),
                    edges=config.get("edges", []),
                    variables=config.get("variables", {}),
                    config=config.get("config", {}),
                    metadata=config.get("metadata"),
                    error_handling=config.get("error_handling"),
                    observability=config.get("observability"),
                    security=config.get("security"),
                    raw_config=config,
                )
                self._workflow_templates[template_id] = template
                logger.debug("Loaded workflow template", template_id=template_id)

            except Exception as e:
                logger.error("Failed to load workflow template", file=str(file_path), error=str(e))

    # =========================================================================
    # Agent Templates
    # =========================================================================

    async def list_agent_templates(self) -> List[TemplateInfo]:
        """List all available agent templates."""
        await self.initialize()

        return [
            TemplateInfo(
                id=t.id,
                name=t.name,
                description=t.description,
                type=TemplateType.AGENT,
                version=t.version,
                category=t.type.value,
            )
            for t in self._agent_templates.values()
        ]

    async def get_agent_template(self, template_id: str) -> Optional[AgentTemplate]:
        """Get a specific agent template by ID."""
        await self.initialize()
        return self._agent_templates.get(template_id)

    async def get_agent_template_config(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get the raw configuration for an agent template."""
        template = await self.get_agent_template(template_id)
        return template.raw_config if template else None

    async def get_agent_enhancements(self, template_id: str) -> Dict[str, Any]:
        """Get available enhancements for an agent template."""
        template = await self.get_agent_template(template_id)
        if not template:
            return {}

        return {
            "enhancements": template.enhancements,
            "capabilities": template.capabilities,
            "description": "Configure these options to customize agent behavior",
        }

    # =========================================================================
    # Workflow Templates
    # =========================================================================

    async def list_workflow_templates(self) -> List[TemplateInfo]:
        """List all available workflow templates."""
        await self.initialize()

        return [
            TemplateInfo(
                id=t.id,
                name=t.name,
                description=t.description,
                type=TemplateType.WORKFLOW,
                version=t.version,
                category=t.metadata.get("category") if t.metadata else None,
                complexity=t.metadata.get("complexity") if t.metadata else None,
                patterns_used=t.metadata.get("patterns_used") if t.metadata else None,
            )
            for t in self._workflow_templates.values()
        ]

    async def get_workflow_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a specific workflow template by ID."""
        await self.initialize()
        return self._workflow_templates.get(template_id)

    async def get_workflow_template_config(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get the raw configuration for a workflow template."""
        template = await self.get_workflow_template(template_id)
        return template.raw_config if template else None

    # =========================================================================
    # Template Instantiation
    # =========================================================================

    async def create_agent_from_template(
        self,
        template_id: str,
        customizations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new agent configuration from a template.

        Args:
            template_id: ID of the template to use
            customizations: Optional customizations to apply

        Returns:
            Agent configuration ready for creation
        """
        template = await self.get_agent_template(template_id)
        if not template:
            raise ValueError(f"Agent template not found: {template_id}")

        # Start with template config
        config = template.raw_config.copy()

        # Apply customizations
        if customizations:
            config = self._deep_merge(config, customizations)

        # Add metadata
        config["_template"] = {
            "id": template_id,
            "version": template.version,
            "created_at": datetime.utcnow().isoformat(),
        }

        return config

    async def create_workflow_from_template(
        self,
        template_id: str,
        customizations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new workflow configuration from a template.

        Args:
            template_id: ID of the template to use
            customizations: Optional customizations to apply

        Returns:
            Workflow configuration ready for creation
        """
        template = await self.get_workflow_template(template_id)
        if not template:
            raise ValueError(f"Workflow template not found: {template_id}")

        # Start with template config
        config = template.raw_config.copy()

        # Apply customizations
        if customizations:
            config = self._deep_merge(config, customizations)

        # Add metadata
        config["_template"] = {
            "id": template_id,
            "version": template.version,
            "created_at": datetime.utcnow().isoformat(),
        }

        return config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


# =============================================================================
# Singleton Instance
# =============================================================================

_template_service: Optional[TemplateService] = None


async def get_template_service() -> TemplateService:
    """Get or create the template service singleton."""
    global _template_service
    if _template_service is None:
        _template_service = TemplateService()
        await _template_service.initialize()
    return _template_service
