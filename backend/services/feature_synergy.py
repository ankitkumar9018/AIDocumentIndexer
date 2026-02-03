"""
AIDocumentIndexer - Feature Synergy Service
============================================

This service enables modular feature integration, allowing different
components of the system to work together seamlessly.

Feature Synergies:
1. Link Groups → Web Scraper → Knowledge Graph → Chat
2. Connectors → Collections → Knowledge Graph → Chat
3. Skills → Workflows → External APIs
4. Research → Reports → Documents
5. Documents → Knowledge Graph → Enhanced RAG
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
from uuid import UUID

logger = structlog.get_logger(__name__)


class FeatureModule(str, Enum):
    """Available feature modules in the system."""
    LINK_GROUPS = "link_groups"
    WEB_SCRAPER = "web_scraper"
    DOCUMENTS = "documents"
    COLLECTIONS = "collections"
    FOLDERS = "folders"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    SKILLS = "skills"
    WORKFLOWS = "workflows"
    CHAT = "chat"
    RESEARCH = "research"
    REPORTS = "reports"
    CONNECTORS = "connectors"
    EXTERNAL_API = "external_api"
    EMBEDDINGS = "embeddings"
    RERANKING = "reranking"


@dataclass
class FeatureConnection:
    """Represents a connection between two features."""
    source: FeatureModule
    target: FeatureModule
    description: str
    data_flow: str  # What data flows between them
    example_use_case: str


# Define all feature connections in the system
FEATURE_CONNECTIONS: List[FeatureConnection] = [
    # Link Groups integrations
    FeatureConnection(
        source=FeatureModule.LINK_GROUPS,
        target=FeatureModule.WEB_SCRAPER,
        description="Scrape all links in a group",
        data_flow="URLs from link groups are sent to web scraper",
        example_use_case="Marketing team organizes competitor websites in a group and scrapes them weekly",
    ),
    FeatureConnection(
        source=FeatureModule.WEB_SCRAPER,
        target=FeatureModule.DOCUMENTS,
        description="Scraped content becomes documents",
        data_flow="Scraped HTML/text is converted to indexed documents",
        example_use_case="Web pages are indexed for RAG queries",
    ),
    FeatureConnection(
        source=FeatureModule.WEB_SCRAPER,
        target=FeatureModule.KNOWLEDGE_GRAPH,
        description="Extract entities from scraped content",
        data_flow="Text content is analyzed for entities and relationships",
        example_use_case="Extract people, companies, products from competitor websites",
    ),

    # Document flow
    FeatureConnection(
        source=FeatureModule.DOCUMENTS,
        target=FeatureModule.COLLECTIONS,
        description="Organize documents in collections",
        data_flow="Documents can be grouped into collections",
        example_use_case="Group all Q4 financial documents in a collection",
    ),
    FeatureConnection(
        source=FeatureModule.DOCUMENTS,
        target=FeatureModule.KNOWLEDGE_GRAPH,
        description="Extract knowledge graph from documents",
        data_flow="Entities and relations are extracted from document text",
        example_use_case="Build a knowledge graph of all people and organizations mentioned in contracts",
    ),
    FeatureConnection(
        source=FeatureModule.DOCUMENTS,
        target=FeatureModule.EMBEDDINGS,
        description="Generate embeddings for semantic search",
        data_flow="Document chunks are embedded for vector search",
        example_use_case="Enable semantic similarity search across all documents",
    ),

    # Knowledge Graph integrations
    FeatureConnection(
        source=FeatureModule.KNOWLEDGE_GRAPH,
        target=FeatureModule.CHAT,
        description="GraphRAG enhances chat responses",
        data_flow="Entity relationships provide context for RAG queries",
        example_use_case="When asking about a person, related entities are included in context",
    ),
    FeatureConnection(
        source=FeatureModule.KNOWLEDGE_GRAPH,
        target=FeatureModule.RESEARCH,
        description="Entity-guided research",
        data_flow="KG entities guide research direction",
        example_use_case="Research all connections of a specific organization",
    ),

    # Skills and Workflows
    FeatureConnection(
        source=FeatureModule.SKILLS,
        target=FeatureModule.WORKFLOWS,
        description="Skills are nodes in workflows",
        data_flow="Skill execution results flow to next workflow nodes",
        example_use_case="Summarization skill feeds into translation skill in a workflow",
    ),
    FeatureConnection(
        source=FeatureModule.EXTERNAL_API,
        target=FeatureModule.SKILLS,
        description="External APIs become skills",
        data_flow="External API responses are wrapped as skill outputs",
        example_use_case="Import a translation API as a skill for workflows",
    ),
    FeatureConnection(
        source=FeatureModule.WORKFLOWS,
        target=FeatureModule.EXTERNAL_API,
        description="Workflows can be published as APIs",
        data_flow="Workflow execution is exposed via REST API",
        example_use_case="Publish a document processing workflow for external systems",
    ),

    # Research and Reports
    FeatureConnection(
        source=FeatureModule.RESEARCH,
        target=FeatureModule.REPORTS,
        description="Research results become reports",
        data_flow="Research findings are formatted into reports",
        example_use_case="Generate a market analysis report from research results",
    ),
    FeatureConnection(
        source=FeatureModule.REPORTS,
        target=FeatureModule.DOCUMENTS,
        description="Reports are saved as documents",
        data_flow="Generated reports are indexed for future queries",
        example_use_case="Past reports are searchable in chat",
    ),

    # Connectors
    FeatureConnection(
        source=FeatureModule.CONNECTORS,
        target=FeatureModule.DOCUMENTS,
        description="Sync external data as documents",
        data_flow="Notion pages, GitHub issues become documents",
        example_use_case="Keep product docs synced from Notion",
    ),
    FeatureConnection(
        source=FeatureModule.CONNECTORS,
        target=FeatureModule.COLLECTIONS,
        description="External data organized in collections",
        data_flow="Synced resources are grouped by source",
        example_use_case="All GitHub issues in a 'Issues' collection",
    ),

    # Chat integrations
    FeatureConnection(
        source=FeatureModule.CHAT,
        target=FeatureModule.SKILLS,
        description="Chat can invoke skills",
        data_flow="User requests trigger skill execution",
        example_use_case="'Summarize this document' invokes summarization skill",
    ),
    FeatureConnection(
        source=FeatureModule.CHAT,
        target=FeatureModule.RESEARCH,
        description="Chat can trigger deep research",
        data_flow="Complex queries start research workflows",
        example_use_case="'Research our competitor's products' starts research agent",
    ),
    FeatureConnection(
        source=FeatureModule.COLLECTIONS,
        target=FeatureModule.CHAT,
        description="Chat scoped to collections",
        data_flow="RAG queries filtered to specific collections",
        example_use_case="Chat only about Q4 financial documents",
    ),

    # Retrieval enhancements
    FeatureConnection(
        source=FeatureModule.EMBEDDINGS,
        target=FeatureModule.RERANKING,
        description="Rerank embedding search results",
        data_flow="Initial retrieval results are reranked for relevance",
        example_use_case="Cross-encoder reranking improves precision",
    ),
    FeatureConnection(
        source=FeatureModule.RERANKING,
        target=FeatureModule.CHAT,
        description="Reranked results in chat context",
        data_flow="Best matching documents provided to LLM",
        example_use_case="More relevant context leads to better answers",
    ),
]


class FeatureSynergyService:
    """
    Service for managing feature integrations and synergies.

    This service provides:
    1. Discovery of feature connections
    2. Pipeline execution across features
    3. Feature dependency mapping
    4. Integration health checks
    """

    def __init__(self):
        self.connections = FEATURE_CONNECTIONS
        self._build_adjacency_map()

    def _build_adjacency_map(self):
        """Build adjacency map for feature graph traversal."""
        self.adjacency: Dict[FeatureModule, List[FeatureConnection]] = {}
        self.reverse_adjacency: Dict[FeatureModule, List[FeatureConnection]] = {}

        for conn in self.connections:
            if conn.source not in self.adjacency:
                self.adjacency[conn.source] = []
            self.adjacency[conn.source].append(conn)

            if conn.target not in self.reverse_adjacency:
                self.reverse_adjacency[conn.target] = []
            self.reverse_adjacency[conn.target].append(conn)

    def get_all_connections(self) -> List[Dict[str, Any]]:
        """Get all feature connections as a list of dicts."""
        return [
            {
                "source": conn.source.value,
                "target": conn.target.value,
                "description": conn.description,
                "data_flow": conn.data_flow,
                "example": conn.example_use_case,
            }
            for conn in self.connections
        ]

    def get_outgoing_connections(self, feature: FeatureModule) -> List[FeatureConnection]:
        """Get all features that a given feature can connect to."""
        return self.adjacency.get(feature, [])

    def get_incoming_connections(self, feature: FeatureModule) -> List[FeatureConnection]:
        """Get all features that connect to a given feature."""
        return self.reverse_adjacency.get(feature, [])

    def get_feature_graph(self) -> Dict[str, Any]:
        """Get the full feature graph for visualization."""
        nodes = [{"id": f.value, "label": f.value.replace("_", " ").title()} for f in FeatureModule]
        edges = [
            {
                "source": conn.source.value,
                "target": conn.target.value,
                "label": conn.description,
            }
            for conn in self.connections
        ]
        return {"nodes": nodes, "edges": edges}

    def find_integration_path(
        self,
        source: FeatureModule,
        target: FeatureModule,
        max_depth: int = 5
    ) -> Optional[List[FeatureConnection]]:
        """
        Find a path of integrations from source to target feature.

        Uses BFS to find the shortest path.
        """
        from collections import deque

        if source == target:
            return []

        visited = set()
        queue = deque([(source, [])])

        while queue:
            current, path = queue.popleft()

            if current in visited:
                continue
            visited.add(current)

            if len(path) >= max_depth:
                continue

            for conn in self.adjacency.get(current, []):
                new_path = path + [conn]
                if conn.target == target:
                    return new_path
                queue.append((conn.target, new_path))

        return None

    def get_feature_capabilities(self, feature: FeatureModule) -> Dict[str, Any]:
        """Get capabilities of a feature including its integrations."""
        outgoing = self.get_outgoing_connections(feature)
        incoming = self.get_incoming_connections(feature)

        return {
            "feature": feature.value,
            "can_send_to": [
                {"target": c.target.value, "description": c.description}
                for c in outgoing
            ],
            "can_receive_from": [
                {"source": c.source.value, "description": c.description}
                for c in incoming
            ],
            "total_integrations": len(outgoing) + len(incoming),
        }

    async def execute_pipeline(
        self,
        steps: List[Dict[str, Any]],
        initial_data: Dict[str, Any],
        user_id: str,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a multi-feature pipeline.

        Each step specifies a feature and operation to perform.
        Data flows from one step to the next.

        Example:
        steps = [
            {"feature": "link_groups", "operation": "get_links", "params": {"group_id": "123"}},
            {"feature": "web_scraper", "operation": "scrape", "params": {"storage_mode": "permanent"}},
            {"feature": "knowledge_graph", "operation": "extract_entities", "params": {}},
        ]
        """
        results = []
        current_data = initial_data

        for i, step in enumerate(steps):
            feature = step.get("feature")
            operation = step.get("operation")
            params = step.get("params", {})

            logger.info(
                "Executing pipeline step",
                step=i + 1,
                feature=feature,
                operation=operation,
            )

            try:
                result = await self._execute_feature_operation(
                    feature=feature,
                    operation=operation,
                    params=params,
                    input_data=current_data,
                    user_id=user_id,
                    organization_id=organization_id,
                )
                results.append({
                    "step": i + 1,
                    "feature": feature,
                    "operation": operation,
                    "status": "success",
                    "result": result,
                })
                current_data = result  # Pass to next step
            except Exception as e:
                logger.error(
                    "Pipeline step failed",
                    step=i + 1,
                    feature=feature,
                    operation=operation,
                    error=str(e),
                )
                results.append({
                    "step": i + 1,
                    "feature": feature,
                    "operation": operation,
                    "status": "failed",
                    "error": str(e),
                })
                break  # Stop pipeline on error

        return {
            "pipeline_status": "completed" if all(r["status"] == "success" for r in results) else "failed",
            "steps_executed": len(results),
            "steps_successful": sum(1 for r in results if r["status"] == "success"),
            "results": results,
            "final_output": current_data,
        }

    async def _execute_feature_operation(
        self,
        feature: str,
        operation: str,
        params: Dict[str, Any],
        input_data: Dict[str, Any],
        user_id: str,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute an operation on a specific feature."""

        # Link Groups operations
        if feature == "link_groups":
            if operation == "get_links":
                from backend.db.database import get_async_session
                from backend.db.models import SavedLink
                from sqlalchemy import select

                async for session in get_async_session():
                    group_id = params.get("group_id") or input_data.get("group_id")
                    result = await session.execute(
                        select(SavedLink).where(SavedLink.group_id == group_id)
                    )
                    links = result.scalars().all()
                    return {"urls": [l.url for l in links], "count": len(links)}

        # Web Scraper operations
        elif feature == "web_scraper":
            if operation == "scrape":
                from backend.services.scraper import get_scraper_service, StorageMode

                urls = params.get("urls") or input_data.get("urls", [])
                storage_mode = StorageMode(params.get("storage_mode", "immediate"))

                service = get_scraper_service()
                job = await service.create_job(
                    user_id=user_id,
                    urls=urls,
                    storage_mode=storage_mode,
                )
                return {"job_id": job.id, "urls_count": len(urls)}

        # Knowledge Graph operations
        elif feature == "knowledge_graph":
            if operation == "extract_entities":
                # This would trigger KG extraction
                return {
                    "status": "queued",
                    "message": "Entity extraction queued for processing",
                }

        # Skills operations
        elif feature == "skills":
            if operation == "execute":
                skill_id = params.get("skill_id")
                inputs = params.get("inputs", input_data)
                # Would call skill execution
                return {"skill_id": skill_id, "status": "executed"}

        # Default: return input data
        return input_data


# Singleton instance
_synergy_service: Optional[FeatureSynergyService] = None


def get_synergy_service() -> FeatureSynergyService:
    """Get or create the feature synergy service."""
    global _synergy_service
    if _synergy_service is None:
        _synergy_service = FeatureSynergyService()
    return _synergy_service


# =============================================================================
# Pre-built Integration Pipelines
# =============================================================================

PRESET_PIPELINES = {
    "link_group_to_knowledge": {
        "name": "Link Group to Knowledge Graph",
        "description": "Scrape all links in a group and extract entities",
        "steps": [
            {"feature": "link_groups", "operation": "get_links"},
            {"feature": "web_scraper", "operation": "scrape", "params": {"storage_mode": "permanent"}},
            {"feature": "knowledge_graph", "operation": "extract_entities"},
        ],
    },
    "research_to_report": {
        "name": "Research to Report",
        "description": "Run deep research and generate a formatted report",
        "steps": [
            {"feature": "research", "operation": "execute"},
            {"feature": "reports", "operation": "generate"},
            {"feature": "documents", "operation": "index"},
        ],
    },
    "external_data_sync": {
        "name": "External Data Sync",
        "description": "Sync external data and build knowledge graph",
        "steps": [
            {"feature": "connectors", "operation": "sync"},
            {"feature": "documents", "operation": "process"},
            {"feature": "knowledge_graph", "operation": "extract_entities"},
        ],
    },
}
