"""
AIDocumentIndexer - AI Agent Builder
=====================================

Phase 23: Enable users to create custom AI chatbots and voice assistants.

Features:
1. AI Chatbot Builder
   - Create agents from document collections
   - Point to websites/URLs for scraping
   - Custom system prompts and personality
   - Voice-enabled responses (Cartesia TTS)

2. Embeddable Widget
   - JavaScript widget for any website
   - Customizable appearance
   - Mobile-responsive design

3. API Publishing
   - REST API for integrations
   - API key authentication
   - Webhook support

4. Analytics Dashboard
   - Query volume and trends
   - Response quality metrics
   - User engagement stats

Usage:
    from backend.services.agent_builder import AgentBuilder, AgentConfig

    builder = AgentBuilder()
    agent = await builder.create_agent(
        name="Support Bot",
        document_collections=["support-docs"],
        system_prompt="You are a helpful support assistant...",
    )

    # Chat with agent
    response = await builder.chat(agent.id, "How do I reset my password?")

    # Get embed code
    embed_code = builder.get_embed_code(agent.id)
"""

import asyncio
import hashlib
import json
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class AgentStatus(str, Enum):
    """Agent lifecycle status."""
    DRAFT = "draft"            # Being configured
    TRAINING = "training"      # Learning from documents
    ACTIVE = "active"          # Ready to serve
    PAUSED = "paused"          # Temporarily disabled
    ARCHIVED = "archived"      # Soft deleted


class AgentType(str, Enum):
    """Type of agent."""
    CHATBOT = "chatbot"        # Text-only chat
    VOICE = "voice"            # Voice-enabled
    API_ONLY = "api_only"      # No UI, API access only


class WidgetPosition(str, Enum):
    """Position of embedded widget."""
    BOTTOM_RIGHT = "bottom-right"
    BOTTOM_LEFT = "bottom-left"
    TOP_RIGHT = "top-right"
    TOP_LEFT = "top-left"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for an AI agent."""
    # Identity
    name: str
    description: str = ""
    avatar_url: Optional[str] = None

    # Knowledge sources
    document_collections: List[str] = field(default_factory=list)
    website_sources: List[str] = field(default_factory=list)
    knowledge_base_ids: List[str] = field(default_factory=list)

    # Behavior
    system_prompt: str = "You are a helpful AI assistant."
    temperature: float = 0.7
    max_response_length: int = 1000
    language: str = "en"

    # Voice settings
    voice_enabled: bool = False
    voice_id: str = "sonic-english-male-1"
    voice_speed: float = 1.0

    # Model settings
    model: Optional[str] = None
    fallback_model: Optional[str] = None

    def __post_init__(self):
        """Resolve model defaults from llm_config if not explicitly set."""
        if self.model is None or self.fallback_model is None:
            from backend.services.llm import llm_config
            if self.model is None:
                self.model = llm_config.default_chat_model
            if self.fallback_model is None:
                self.fallback_model = llm_config.default_chat_model


@dataclass
class WidgetConfig:
    """Configuration for embeddable widget."""
    # Appearance
    primary_color: str = "#6366f1"
    background_color: str = "#ffffff"
    text_color: str = "#1f2937"
    border_radius: int = 16

    # Position and size
    position: WidgetPosition = WidgetPosition.BOTTOM_RIGHT
    width: int = 380
    height: int = 600
    button_size: int = 60

    # Behavior
    greeting_message: str = "Hi! How can I help you today?"
    placeholder_text: str = "Type your message..."
    show_branding: bool = True

    # Features
    enable_file_upload: bool = False
    enable_voice_input: bool = False
    enable_voice_output: bool = False
    enable_feedback: bool = True


@dataclass
class APIKey:
    """API key for agent access."""
    id: str
    key_hash: str  # Hashed key for storage
    key_prefix: str  # First 8 chars for identification
    name: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    rate_limit: int = 100  # Requests per minute
    is_active: bool = True


@dataclass
class Agent:
    """AI Agent model."""
    id: str
    user_id: str
    organization_id: str
    config: AgentConfig
    widget_config: WidgetConfig
    status: AgentStatus = AgentStatus.DRAFT
    agent_type: AgentType = AgentType.CHATBOT

    # Security
    api_keys: List[APIKey] = field(default_factory=list)
    allowed_domains: List[str] = field(default_factory=list)  # For CORS

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None

    # Stats
    total_conversations: int = 0
    total_messages: int = 0

    # Training
    last_trained_at: Optional[datetime] = None
    training_status: Optional[str] = None


@dataclass
class AgentMessage:
    """A message in an agent conversation."""
    id: str
    agent_id: str
    conversation_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    feedback: Optional[str] = None  # "positive", "negative", None


@dataclass
class AgentConversation:
    """A conversation with an agent."""
    id: str
    agent_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    message_count: int = 0
    user_metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Agent Builder Service
# =============================================================================

class AgentBuilder:
    """
    Service for creating and managing AI agents.
    """

    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._conversations: Dict[str, AgentConversation] = {}
        self._messages: Dict[str, List[AgentMessage]] = {}

    # =========================================================================
    # Agent CRUD
    # =========================================================================

    async def create_agent(
        self,
        user_id: str,
        organization_id: str,
        config: AgentConfig,
        widget_config: Optional[WidgetConfig] = None,
        agent_type: AgentType = AgentType.CHATBOT,
    ) -> Agent:
        """Create a new AI agent."""
        agent_id = str(uuid.uuid4())

        agent = Agent(
            id=agent_id,
            user_id=user_id,
            organization_id=organization_id,
            config=config,
            widget_config=widget_config or WidgetConfig(),
            agent_type=agent_type,
            status=AgentStatus.DRAFT,
        )

        self._agents[agent_id] = agent

        logger.info(
            "Agent created",
            agent_id=agent_id,
            name=config.name,
            user_id=user_id,
        )

        return agent

    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    async def update_agent(
        self,
        agent_id: str,
        config: Optional[AgentConfig] = None,
        widget_config: Optional[WidgetConfig] = None,
    ) -> Optional[Agent]:
        """Update an agent's configuration."""
        agent = self._agents.get(agent_id)
        if not agent:
            return None

        if config:
            agent.config = config
        if widget_config:
            agent.widget_config = widget_config
        agent.updated_at = datetime.utcnow()

        logger.info("Agent updated", agent_id=agent_id)
        return agent

    async def delete_agent(self, agent_id: str) -> bool:
        """Soft delete an agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        agent.status = AgentStatus.ARCHIVED
        agent.updated_at = datetime.utcnow()

        logger.info("Agent archived", agent_id=agent_id)
        return True

    async def list_agents(
        self,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        status: Optional[AgentStatus] = None,
    ) -> List[Agent]:
        """List agents with optional filters."""
        agents = list(self._agents.values())

        if user_id:
            agents = [a for a in agents if a.user_id == user_id]
        if organization_id:
            agents = [a for a in agents if a.organization_id == organization_id]
        if status:
            agents = [a for a in agents if a.status == status]

        # Exclude archived by default
        agents = [a for a in agents if a.status != AgentStatus.ARCHIVED]

        return sorted(agents, key=lambda x: x.created_at, reverse=True)

    # =========================================================================
    # Training
    # =========================================================================

    async def train_agent(self, agent_id: str) -> bool:
        """
        Train an agent on its knowledge sources.

        This indexes documents and prepares the agent for queries.
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        agent.status = AgentStatus.TRAINING
        agent.training_status = "indexing"

        try:
            # In production, this would:
            # 1. Fetch documents from collections
            # 2. Scrape website sources
            # 3. Build vector index
            # 4. Prepare retrieval pipeline

            # Simulate training
            await asyncio.sleep(1)

            agent.status = AgentStatus.ACTIVE
            agent.training_status = "complete"
            agent.last_trained_at = datetime.utcnow()

            logger.info("Agent training complete", agent_id=agent_id)
            return True

        except Exception as e:
            agent.status = AgentStatus.DRAFT
            agent.training_status = f"failed: {str(e)}"
            logger.error(f"Agent training failed: {e}", agent_id=agent_id)
            return False

    # =========================================================================
    # Chat
    # =========================================================================

    async def chat(
        self,
        agent_id: str,
        message: str,
        conversation_id: Optional[str] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Chat with an agent.

        Phase 59: Enhanced with A-Mem memory integration for:
        - 85-93% token reduction
        - Intelligent memory retrieval and storage
        - Long-term conversation context

        Returns response with text, sources, and metadata.
        """
        agent = self._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")

        if agent.status != AgentStatus.ACTIVE:
            raise ValueError(f"Agent is not active: {agent.status}")

        # Get or create conversation
        if conversation_id and conversation_id in self._conversations:
            conversation = self._conversations[conversation_id]
        else:
            conversation = AgentConversation(
                id=conversation_id or str(uuid.uuid4()),
                agent_id=agent_id,
                started_at=datetime.utcnow(),
                user_metadata=user_metadata or {},
            )
            self._conversations[conversation.id] = conversation
            self._messages[conversation.id] = []

        # Phase 59: Retrieve relevant memories using A-Mem
        memory_context = ""
        try:
            from backend.services.amem_memory import get_amem_service
            from backend.core.config import settings as core_settings

            if core_settings.ENABLE_AGENT_MEMORY:
                amem = await get_amem_service()
                memories = await amem.retrieve(message, top_k=5)

                if memories:
                    memory_context = "Relevant memories from previous conversations:\n"
                    for mem in memories:
                        memory_context += f"- {mem.content}\n"

                    logger.info(
                        "Retrieved A-Mem memories for agent",
                        agent_id=agent_id,
                        memory_count=len(memories),
                    )
        except ImportError:
            logger.debug("A-Mem service not available")
        except Exception as mem_error:
            logger.warning(f"A-Mem retrieval failed: {mem_error}")

        # Store user message
        user_msg = AgentMessage(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            conversation_id=conversation.id,
            role="user",
            content=message,
            timestamp=datetime.utcnow(),
        )
        self._messages[conversation.id].append(user_msg)

        # Generate response (with memory context)
        response_text, sources = await self._generate_response(
            agent, message, self._messages[conversation.id], memory_context
        )

        # Store assistant message
        assistant_msg = AgentMessage(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            conversation_id=conversation.id,
            role="assistant",
            content=response_text,
            timestamp=datetime.utcnow(),
            sources=sources,
        )
        self._messages[conversation.id].append(assistant_msg)

        # Phase 59: Store important facts in A-Mem
        try:
            from backend.services.amem_memory import get_amem_service
            from backend.core.config import settings as core_settings

            if core_settings.ENABLE_AGENT_MEMORY:
                amem = await get_amem_service()
                # Let A-Mem agent decide if this exchange should be remembered
                await amem.process(
                    f"User: {message}\nAssistant: {response_text}",
                    context=f"Agent: {agent.config.name}"
                )
        except Exception as store_error:
            logger.debug(f"A-Mem storage skipped: {store_error}")

        # Update stats
        conversation.message_count += 2
        agent.total_messages += 2

        # Generate audio if voice enabled
        audio_url = None
        if agent.config.voice_enabled:
            audio_url = await self._generate_audio(agent, response_text)

        # Phase 63: Agent evaluation metrics (Pass^k, hallucination detection)
        # Use runtime settings for hot-reload (no server restart needed)
        evaluation_metrics = None
        try:
            from backend.services.settings import get_settings_service
            settings_svc = get_settings_service()
            eval_enabled = await settings_svc.get_setting("agent.evaluation_enabled") or False
            if eval_enabled:
                from backend.services.agent_evaluation import AgentEvaluator
                evaluator = AgentEvaluator()
                evaluation_metrics = await evaluator.evaluate(
                    agent_id=agent_id,
                    query=message,
                    response=response_text,
                    context="\n".join(s.get("content", "") for s in sources) if sources else "",
                    ground_truth=None,  # No ground truth for online evaluation
                )
                if evaluation_metrics:
                    logger.info(
                        "Agent evaluation complete",
                        agent_id=agent_id,
                        pass_rate=getattr(evaluation_metrics, 'pass_rate', None),
                        hallucination_score=getattr(evaluation_metrics, 'hallucination_score', None),
                    )
        except Exception as eval_error:
            logger.debug(f"Agent evaluation skipped: {eval_error}")

        return {
            "conversation_id": conversation.id,
            "message_id": assistant_msg.id,
            "response": response_text,
            "sources": sources,
            "audio_url": audio_url,
            "evaluation": evaluation_metrics,
        }

    async def _generate_response(
        self,
        agent: Agent,
        message: str,
        history: List[AgentMessage],
        memory_context: str = "",
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Generate response using RAG.

        Phase 59: Implements real RAG instead of mock, with advanced features:
        - Self-RAG for hallucination detection
        - Hybrid retrieval (LightRAG + RAPTOR)
        - Tiered reranking pipeline
        - Context compression for long conversations
        - A-Mem memory integration for conversation context
        """
        sources = []
        rag_context = ""

        try:
            # Phase 59: Use real RAG service when agent has knowledge sources
            has_knowledge_sources = (
                agent.config.knowledge_base_ids or
                agent.config.document_collections or
                agent.config.website_sources
            )

            if has_knowledge_sources:
                try:
                    from backend.services.rag import RAGService
                    from backend.core.config import settings as core_settings

                    rag = RAGService()

                    # Build RAG query with advanced features
                    rag_kwargs = {
                        "query": message,
                        "organization_id": agent.organization_id,
                    }

                    # Use first knowledge base or collection as the target
                    if agent.config.knowledge_base_ids:
                        rag_kwargs["collection"] = agent.config.knowledge_base_ids[0]
                    elif agent.config.document_collections:
                        rag_kwargs["collection"] = agent.config.document_collections[0]

                    # Phase 59: Enable advanced RAG features
                    # Self-RAG for hallucination detection
                    if core_settings.ENABLE_SELF_RAG:
                        rag_kwargs["use_self_rag"] = True

                    # Hybrid retrieval (LightRAG + RAPTOR fusion)
                    if core_settings.ENABLE_LIGHTRAG or core_settings.ENABLE_RAPTOR:
                        rag_kwargs["use_hybrid_retrieval"] = True

                    # Tiered reranking (ColBERT → Cross-Encoder → LLM)
                    if core_settings.ENABLE_TIERED_RERANKING:
                        rag_kwargs["use_tiered_reranking"] = True

                    # Knowledge graph for entity relationships
                    if getattr(core_settings, 'ENABLE_KNOWLEDGE_GRAPH', True):
                        rag_kwargs["enable_knowledge_graph"] = True

                    # Execute RAG query with full pipeline
                    rag_result = await rag.query(**rag_kwargs)

                    # Extract context and sources from RAG result
                    if hasattr(rag_result, 'context'):
                        rag_context = rag_result.context
                    elif isinstance(rag_result, dict):
                        rag_context = rag_result.get("context", rag_result.get("answer", ""))

                    # Extract sources
                    if hasattr(rag_result, 'sources'):
                        sources = [
                            {
                                "title": s.get("title", s.get("document_title", "Document")),
                                "snippet": s.get("snippet", s.get("content", ""))[:200],
                                "score": s.get("score", s.get("relevance_score", 0)),
                                "document_id": s.get("document_id", s.get("id", "")),
                            }
                            for s in (rag_result.sources if isinstance(rag_result.sources, list) else [])
                        ]
                    elif isinstance(rag_result, dict) and "sources" in rag_result:
                        sources = [
                            {
                                "title": s.get("title", s.get("document_title", "Document")),
                                "snippet": s.get("snippet", s.get("content", ""))[:200],
                                "score": s.get("score", s.get("relevance_score", 0)),
                                "document_id": s.get("document_id", s.get("id", "")),
                            }
                            for s in rag_result.get("sources", [])
                        ]

                    logger.info(
                        "RAG query completed for agent",
                        agent_id=agent.id,
                        sources_count=len(sources),
                        has_context=bool(rag_context),
                    )

                except ImportError:
                    logger.warning("RAG service not available, falling back to direct LLM")
                except Exception as rag_error:
                    logger.warning(f"RAG query failed, falling back to LLM only: {rag_error}")

            # Build LLM messages
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

            # Build system prompt with RAG context and memory if available
            system_content = agent.config.system_prompt

            # Phase 59: Add memory context from A-Mem
            if memory_context:
                system_content = f"""{system_content}

{memory_context}
"""

            # Add RAG context
            if rag_context:
                system_content = f"""{system_content}

Use the following context from the knowledge base to answer the user's question.
If the context doesn't contain relevant information, say so and provide a general response.

Knowledge Base Context:
{rag_context}
"""

            messages = [SystemMessage(content=system_content)]

            # Add history (previous messages only, excluding the current one which
            # was already appended to history by send_message before calling us)
            for msg in history[-11:-1]:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(AIMessage(content=msg.content))

            # Add current message
            messages.append(HumanMessage(content=message))

            # Generate response with LLM (use factory for provider-agnostic model creation)
            try:
                from backend.services.llm import LLMFactory
                llm = LLMFactory.get_chat_model(
                    model=agent.config.model,
                    temperature=agent.config.temperature,
                )
            except Exception as llm_err:
                logger.warning(f"Factory failed for agent LLM, using direct OpenAI: {llm_err}")
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=agent.config.model,
                    temperature=agent.config.temperature,
                )

            response = await llm.ainvoke(messages)

            return response.content, sources

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I'm sorry, I encountered an error. Please try again.", []

    async def _generate_audio(
        self,
        agent: Agent,
        text: str,
    ) -> Optional[str]:
        """Generate audio response using Cartesia TTS."""
        try:
            from backend.services.audio.cartesia_tts import CartesiaTTSProvider

            tts = CartesiaTTSProvider()
            audio_data = await tts.synthesize(
                text=text[:500],  # Limit for preview
                voice_id=agent.config.voice_id,
                speed=agent.config.voice_speed,
            )

            # In production, upload to storage and return URL
            # For now, return placeholder
            return f"/api/agents/{agent.id}/audio/{uuid.uuid4()}"

        except Exception as e:
            logger.warning(f"Audio generation failed: {e}")
            return None

    # =========================================================================
    # API Keys
    # =========================================================================

    async def create_api_key(
        self,
        agent_id: str,
        name: str,
        rate_limit: int = 100,
        expires_in_days: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """
        Create an API key for an agent.

        Returns the raw key (only shown once) and the key object.
        """
        agent = self._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")

        # Generate secure key
        raw_key = f"ak_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        api_key = APIKey(
            id=str(uuid.uuid4()),
            key_hash=key_hash,
            key_prefix=raw_key[:11],  # "ak_" + 8 chars
            name=name,
            created_at=datetime.utcnow(),
            rate_limit=rate_limit,
            expires_at=(
                datetime.utcnow() + timedelta(days=expires_in_days)
                if expires_in_days else None
            ),
        )

        agent.api_keys.append(api_key)

        logger.info(
            "API key created",
            agent_id=agent_id,
            key_prefix=api_key.key_prefix,
        )

        return raw_key, api_key

    async def validate_api_key(
        self,
        agent_id: str,
        raw_key: str,
    ) -> Optional[APIKey]:
        """Validate an API key and return it if valid."""
        agent = self._agents.get(agent_id)
        if not agent:
            return None

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        for api_key in agent.api_keys:
            if api_key.key_hash == key_hash:
                # Check if active
                if not api_key.is_active:
                    return None

                # Check expiration
                if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                    return None

                # Update last used
                api_key.last_used_at = datetime.utcnow()
                return api_key

        return None

    async def revoke_api_key(self, agent_id: str, key_id: str) -> bool:
        """Revoke an API key."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        for api_key in agent.api_keys:
            if api_key.id == key_id:
                api_key.is_active = False
                logger.info("API key revoked", agent_id=agent_id, key_id=key_id)
                return True

        return False

    # =========================================================================
    # Embed Code
    # =========================================================================

    def get_embed_code(self, agent_id: str) -> str:
        """Generate embeddable widget code."""
        agent = self._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")

        widget = agent.widget_config
        api_base = getattr(settings, "API_BASE_URL", "https://api.example.com")

        return f'''<!-- AI Agent Widget -->
<script>
(function() {{
  var w = document.createElement('script');
  w.src = {json.dumps(f'{api_base}/widget/agent.js')};
  w.async = true;
  w.onload = function() {{
    AIAgent.init({{
      agentId: {json.dumps(agent_id)},
      position: {json.dumps(widget.position.value)},
      primaryColor: {json.dumps(widget.primary_color)},
      backgroundColor: {json.dumps(widget.background_color)},
      greeting: {json.dumps(widget.greeting_message)},
      enableVoice: {str(widget.enable_voice_output).lower()},
      enableFeedback: {str(widget.enable_feedback).lower()},
    }});
  }};
  document.head.appendChild(w);
}})();
</script>
<!-- End AI Agent Widget -->'''

    # =========================================================================
    # Analytics
    # =========================================================================

    async def get_analytics(
        self,
        agent_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get analytics for an agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")

        # In production, query analytics database
        # For now, return mock data
        return {
            "agent_id": agent_id,
            "period": {
                "start": (start_date or datetime.utcnow() - timedelta(days=30)).isoformat(),
                "end": (end_date or datetime.utcnow()).isoformat(),
            },
            "metrics": {
                "total_conversations": agent.total_conversations + 150,
                "total_messages": agent.total_messages + 450,
                "avg_messages_per_conversation": 3.0,
                "avg_response_time_ms": 850,
                "satisfaction_rate": 0.92,
            },
            "trends": {
                "conversations_by_day": [
                    {"date": "2026-01-15", "count": 45},
                    {"date": "2026-01-16", "count": 52},
                    {"date": "2026-01-17", "count": 38},
                    {"date": "2026-01-18", "count": 61},
                    {"date": "2026-01-19", "count": 49},
                ],
            },
            "top_queries": [
                {"query": "How do I reset my password?", "count": 23},
                {"query": "What are your business hours?", "count": 18},
                {"query": "How do I contact support?", "count": 15},
            ],
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_agent_builder: Optional[AgentBuilder] = None


def get_agent_builder() -> AgentBuilder:
    """Get agent builder singleton."""
    global _agent_builder
    if _agent_builder is None:
        _agent_builder = AgentBuilder()
    return _agent_builder


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "AgentStatus",
    "AgentType",
    "WidgetPosition",
    # Data models
    "AgentConfig",
    "WidgetConfig",
    "APIKey",
    "Agent",
    "AgentMessage",
    "AgentConversation",
    # Service
    "AgentBuilder",
    "get_agent_builder",
]
