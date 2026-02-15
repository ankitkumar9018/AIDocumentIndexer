"""
AIDocumentIndexer - Skills API Routes
=====================================

Endpoints for AI-powered skills execution with database persistence.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.services.llm import LLMFactory, llm_config
from backend.services.llm_provider import LLMProviderService
from backend.core.config import settings
from backend.db.database import get_async_session
from backend.db.models import Skill, SkillExecution, SkillExecutionStatus
from backend.api.middleware.auth import AuthenticatedUser


def _to_uuid(user_id: str) -> UUID:
    """Convert user_id string to UUID safely."""
    return UUID(user_id) if isinstance(user_id, str) else user_id

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class SkillInput(BaseModel):
    """Skill input definition."""
    name: str
    type: str = "text"
    description: Optional[str] = None
    required: bool = True
    default: Optional[Any] = None


class SkillOutput(BaseModel):
    """Skill output definition."""
    name: str
    type: str = "text"
    description: Optional[str] = None


class SkillExecuteRequest(BaseModel):
    """Request to execute a skill."""
    skill_id: str = Field(..., description="ID of the skill to execute (UUID or skill_key)")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values for the skill")
    provider_id: Optional[str] = Field(None, description="LLM provider ID to use")
    model: Optional[str] = Field(None, description="Model name to use")


class SkillExecuteResponse(BaseModel):
    """Response from skill execution."""
    output: Any = Field(..., description="Skill output")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    model_used: str = Field(..., description="Model that was used")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")
    execution_id: Optional[str] = Field(None, description="ID of the execution record")


class SkillCreateRequest(BaseModel):
    """Request to create a custom skill."""
    id: Optional[str] = Field(None, description="Unique skill key (auto-generated if not provided)")
    name: str = Field(..., description="Skill name")
    description: str = Field(..., description="Skill description")
    category: str = Field(default="custom", description="Skill category")
    system_prompt: str = Field(..., max_length=100000, description="System prompt for the skill")
    inputs: List[SkillInput] = Field(default_factory=list, max_length=50, description="Input schema")
    outputs: List[SkillOutput] = Field(default_factory=list, max_length=50, description="Output schema")
    tags: List[str] = Field(default_factory=list, max_length=20, description="Skill tags")
    icon: str = Field(default="zap", max_length=50, description="Icon name")
    is_public: bool = Field(default=False, description="Make skill public")


class SkillResponse(BaseModel):
    """Skill response model."""
    id: str
    skill_key: str
    name: str
    description: Optional[str]
    category: str
    icon: str
    tags: List[str]
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    is_public: bool
    is_builtin: bool
    version: str
    use_count: int
    avg_execution_time_ms: Optional[int]
    created_at: datetime
    updated_at: datetime


class SkillCreateResponse(BaseModel):
    """Response from creating a skill."""
    id: str
    skill_key: str
    name: str
    message: str


class SkillImportRequest(BaseModel):
    """Request to import a skill."""
    skill_data: Dict[str, Any] = Field(..., description="Skill definition JSON")


class SkillUpdateRequest(BaseModel):
    """Request to update a skill."""
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    system_prompt: Optional[str] = None
    inputs: Optional[List[SkillInput]] = None
    outputs: Optional[List[SkillOutput]] = None
    tags: Optional[List[str]] = None
    icon: Optional[str] = None
    is_public: Optional[bool] = None
    is_active: Optional[bool] = None


# =============================================================================
# External Agent Models
# =============================================================================

class ExternalAgentAuthType(str):
    """Authentication types for external agents."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"


class ExternalAgentAuth(BaseModel):
    """Authentication configuration for external agents."""
    type: str = Field(default="api_key", description="Authentication type")
    api_key: Optional[str] = Field(None, description="API key for api_key auth")
    api_key_header: Optional[str] = Field(default="X-API-Key", description="Header name for API key")
    bearer_token: Optional[str] = Field(None, description="Bearer token for bearer auth")
    username: Optional[str] = Field(None, description="Username for basic auth")
    password: Optional[str] = Field(None, description="Password for basic auth")
    oauth2_client_id: Optional[str] = Field(None, description="OAuth2 client ID")
    oauth2_client_secret: Optional[str] = Field(None, description="OAuth2 client secret")
    oauth2_token_url: Optional[str] = Field(None, description="OAuth2 token URL")
    custom_headers: Optional[Dict[str, str]] = Field(None, description="Custom headers")


class ExternalAgentInputMapping(BaseModel):
    """Mapping from skill inputs to external API request."""
    skill_input: str = Field(..., description="Name of the skill input")
    api_path: str = Field(..., description="JSON path in request body (e.g., 'data.prompt')")
    transform: Optional[str] = Field(None, description="Optional transformation: stringify, parse_json")


class ExternalAgentOutputMapping(BaseModel):
    """Mapping from external API response to skill outputs."""
    api_path: str = Field(..., description="JSON path in response (e.g., 'result.text')")
    skill_output: str = Field(..., description="Name of the skill output")
    transform: Optional[str] = Field(None, description="Optional transformation")


class ExternalAgentCreateRequest(BaseModel):
    """Request to create a skill from an external agent/API."""
    # Basic info
    name: str = Field(..., description="Name for the skill")
    description: str = Field(..., description="Description of what the external agent does")
    category: str = Field(default="integration", description="Skill category")
    icon: str = Field(default="external-link", description="Icon name")
    tags: List[str] = Field(default_factory=list, description="Tags")

    # External API configuration
    endpoint_url: str = Field(..., description="External API endpoint URL")
    method: str = Field(default="POST", description="HTTP method")
    content_type: str = Field(default="application/json", description="Content type")

    # Authentication
    auth: ExternalAgentAuth = Field(default_factory=ExternalAgentAuth, description="Authentication config")

    # Input/Output mapping
    inputs: List[SkillInput] = Field(..., description="Skill inputs (what user provides)")
    outputs: List[SkillOutput] = Field(default_factory=list, description="Skill outputs")
    input_mapping: List[ExternalAgentInputMapping] = Field(..., description="Map inputs to API request")
    output_mapping: List[ExternalAgentOutputMapping] = Field(default_factory=list, description="Map API response to outputs")

    # Request template
    request_template: Optional[Dict[str, Any]] = Field(None, description="Base request body template")

    # Advanced settings
    timeout_seconds: int = Field(default=60, ge=1, le=300, description="Request timeout")
    retry_count: int = Field(default=1, ge=0, le=5, description="Number of retries on failure")
    rate_limit_per_minute: int = Field(default=60, ge=1, le=600, description="Rate limit for this agent")
    is_public: bool = Field(default=False, description="Make skill public")


class ExternalAgentTestRequest(BaseModel):
    """Request to test an external agent configuration."""
    endpoint_url: str
    method: str = "POST"
    auth: ExternalAgentAuth
    request_body: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 30


class ExternalAgentTestResponse(BaseModel):
    """Response from testing external agent."""
    success: bool
    status_code: Optional[int]
    response_body: Optional[Any]
    error: Optional[str]
    latency_ms: int


# =============================================================================
# Built-in Skill Prompts (for seeding database)
# =============================================================================

BUILTIN_SKILLS = {
    "summarizer": {
        "name": "Document Summarizer",
        "description": "Create concise summaries of documents with configurable length",
        "category": "analysis",
        "icon": "file-text",
        "tags": ["summarization", "analysis", "documents"],
        "system_prompt": """You are an expert document summarizer. Your task is to create a {length} summary of the provided text.

Guidelines:
- For 'short': 2-3 sentences capturing the main point
- For 'medium': 4-6 sentences covering key points
- For 'long': 8-10 sentences with supporting details

Maintain the original tone and focus on the most important information.

Text to summarize:
{content}""",
        "inputs": [
            {"name": "content", "type": "text", "description": "Text to summarize", "required": True},
            {"name": "length", "type": "select", "description": "Summary length", "required": False, "default": "medium"},
        ],
        "outputs": [
            {"name": "summary", "type": "text", "description": "Generated summary"},
        ],
    },
    "fact-checker": {
        "name": "Fact Checker",
        "description": "Analyze text for factual claims and assess their verifiability",
        "category": "validation",
        "icon": "check-circle",
        "tags": ["validation", "facts", "analysis"],
        "system_prompt": """You are a fact-checking expert. Analyze the following text and identify factual claims.

For each claim:
1. Identify the specific claim
2. Assess its verifiability (verifiable, opinion, uncertain)
3. Note any potential issues or red flags

Respond in JSON format:
{{
  "claims_found": number,
  "verified": number,
  "unverified": number,
  "claims": [
    {{
      "claim": "the claim text",
      "verdict": "verifiable|opinion|uncertain",
      "confidence": 0.0-1.0,
      "notes": "any relevant notes"
    }}
  ],
  "analysis": "overall analysis"
}}

Text to analyze:
{content}""",
        "inputs": [
            {"name": "content", "type": "text", "description": "Text to fact-check", "required": True},
        ],
        "outputs": [
            {"name": "claims", "type": "json", "description": "Analyzed claims"},
        ],
    },
    "translator": {
        "name": "Translator",
        "description": "Translate text to different languages while preserving meaning and tone",
        "category": "transformation",
        "icon": "globe",
        "tags": ["translation", "language", "transformation"],
        "system_prompt": """You are an expert translator. Translate the following text to {target_language}.

Requirements:
- Maintain the original meaning and tone
- Preserve formatting where possible
- Handle idioms appropriately for the target language

Text to translate:
{content}""",
        "inputs": [
            {"name": "content", "type": "text", "description": "Text to translate", "required": True},
            {"name": "target_language", "type": "text", "description": "Target language", "required": True},
        ],
        "outputs": [
            {"name": "translation", "type": "text", "description": "Translated text"},
        ],
    },
    "sentiment": {
        "name": "Sentiment Analyzer",
        "description": "Analyze the sentiment and emotions in text",
        "category": "analysis",
        "icon": "heart",
        "tags": ["sentiment", "emotions", "analysis"],
        "system_prompt": """You are a sentiment analysis expert. Analyze the sentiment of the following text.

Respond in JSON format:
{{
  "overall_sentiment": "positive|negative|neutral|mixed",
  "confidence": 0.0-1.0,
  "emotions": {{
    "joy": 0.0-1.0,
    "sadness": 0.0-1.0,
    "anger": 0.0-1.0,
    "fear": 0.0-1.0,
    "surprise": 0.0-1.0
  }},
  "key_phrases": ["phrase1", "phrase2"],
  "analysis": "detailed analysis"
}}

Text to analyze:
{content}""",
        "inputs": [
            {"name": "content", "type": "text", "description": "Text to analyze", "required": True},
        ],
        "outputs": [
            {"name": "sentiment", "type": "json", "description": "Sentiment analysis results"},
        ],
    },
    "entity-extractor": {
        "name": "Entity Extractor",
        "description": "Extract named entities (people, organizations, locations, etc.) from text",
        "category": "extraction",
        "icon": "tag",
        "tags": ["extraction", "NER", "entities"],
        "system_prompt": """You are an entity extraction expert. Extract named entities from the following text.

Entity types to identify:
- PERSON: People's names
- ORGANIZATION: Companies, agencies, institutions
- LOCATION: Places, addresses, countries
- DATE: Dates and time expressions
- MONEY: Monetary values
- PRODUCT: Products and services
- EVENT: Named events

Respond in JSON format:
{{
  "entities": [
    {{
      "text": "entity text",
      "type": "ENTITY_TYPE",
      "confidence": 0.0-1.0
    }}
  ],
  "entity_counts": {{
    "PERSON": number,
    "ORGANIZATION": number,
    ...
  }}
}}

Text to analyze:
{content}""",
        "inputs": [
            {"name": "content", "type": "text", "description": "Text to analyze", "required": True},
        ],
        "outputs": [
            {"name": "entities", "type": "json", "description": "Extracted entities"},
        ],
    },
    "document-comparison": {
        "name": "Document Comparison",
        "description": "Compare two documents to identify similarities and differences",
        "category": "analysis",
        "icon": "git-compare",
        "tags": ["comparison", "diff", "analysis"],
        "system_prompt": """You are a document comparison expert. Compare the following two documents and identify:
1. Key similarities
2. Key differences
3. Content unique to each document

Document 1:
{document1}

Document 2:
{document2}

Respond in JSON format:
{{
  "similarity_score": 0.0-1.0,
  "similarities": ["similarity1", "similarity2"],
  "differences": ["difference1", "difference2"],
  "unique_to_doc1": ["item1", "item2"],
  "unique_to_doc2": ["item1", "item2"],
  "summary": "overall comparison summary"
}}""",
        "inputs": [
            {"name": "document1", "type": "text", "description": "First document", "required": True},
            {"name": "document2", "type": "text", "description": "Second document", "required": True},
        ],
        "outputs": [
            {"name": "comparison", "type": "json", "description": "Comparison results"},
        ],
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

async def get_skill_by_id_or_key(
    db: AsyncSession,
    skill_identifier: str,
    user_id: str,
) -> Optional[Skill]:
    """Get skill by UUID or skill_key."""
    user_uuid = _to_uuid(user_id)

    # Try as UUID first
    try:
        skill_uuid = UUID(skill_identifier)
        result = await db.execute(
            select(Skill).where(
                Skill.id == skill_uuid,
                or_(
                    Skill.user_id == user_uuid,
                    Skill.is_public == True,
                    Skill.is_builtin == True,
                ),
            )
        )
        skill = result.scalar_one_or_none()
        if skill:
            return skill
    except ValueError:
        pass

    # Try as skill_key
    result = await db.execute(
        select(Skill).where(
            Skill.skill_key == skill_identifier,
            or_(
                Skill.user_id == user_uuid,
                Skill.is_public == True,
                Skill.is_builtin == True,
            ),
        )
    )
    return result.scalar_one_or_none()


async def ensure_builtin_skills(db: AsyncSession) -> None:
    """Ensure built-in skills exist in database."""
    for skill_key, skill_data in BUILTIN_SKILLS.items():
        result = await db.execute(
            select(Skill).where(Skill.skill_key == skill_key, Skill.is_builtin == True).limit(1)
        )
        if not result.scalars().first():
            skill = Skill(
                skill_key=skill_key,
                name=skill_data["name"],
                description=skill_data["description"],
                category=skill_data["category"],
                icon=skill_data["icon"],
                tags=skill_data["tags"],
                system_prompt=skill_data["system_prompt"],
                inputs=skill_data["inputs"],
                outputs=skill_data["outputs"],
                is_builtin=True,
                is_public=True,
                is_active=True,
            )
            db.add(skill)
    await db.commit()


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/execute", response_model=SkillExecuteResponse)
async def execute_skill(
    request: SkillExecuteRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Execute an AI skill with the specified inputs.

    Uses the configured LLM provider to run the skill.
    """
    start_time = datetime.utcnow()

    logger.info(
        "Executing skill",
        skill_id=request.skill_id,
        user_id=user.user_id,
        provider_id=request.provider_id,
    )

    # Ensure built-in skills exist
    await ensure_builtin_skills(db)

    # Get skill from database
    skill = await get_skill_by_id_or_key(db, request.skill_id, user.user_id)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {request.skill_id}"
        )

    if not skill.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This skill is currently disabled"
        )

    user_uuid = _to_uuid(user.user_id)

    # Create execution record
    execution = SkillExecution(
        skill_id=skill.id,
        user_id=user_uuid,
        status=SkillExecutionStatus.RUNNING.value,
        inputs=request.inputs,
    )
    db.add(execution)
    await db.flush()

    # Get provider configuration — use system default, not hardcoded "openai"
    provider_type = llm_config.default_provider
    model_name = request.model or None

    if request.provider_id:
        try:
            provider = await LLMProviderService.get_provider(db, request.provider_id)
            if provider:
                provider_type = provider.provider_type
                model_name = request.model or provider.default_chat_model
        except Exception as e:
            logger.warning(f"Could not load provider {request.provider_id}: {e}")

    # Format the prompt with inputs
    try:
        format_kwargs = {"skill_id": request.skill_id, "inputs": str(request.inputs)}

        # Map common input names
        if "document" in request.inputs:
            doc = request.inputs["document"]
            format_kwargs["content"] = doc.get("content", "") if isinstance(doc, dict) else str(doc)
        if "text" in request.inputs:
            format_kwargs["content"] = request.inputs["text"]
        if "content" in request.inputs:
            format_kwargs["content"] = request.inputs["content"]
        if "length" in request.inputs:
            format_kwargs["length"] = request.inputs["length"]
        if "target_language" in request.inputs:
            format_kwargs["target_language"] = request.inputs["target_language"]
        if "document1" in request.inputs:
            doc1 = request.inputs["document1"]
            format_kwargs["document1"] = doc1.get("content", "") if isinstance(doc1, dict) else str(doc1)
        if "document2" in request.inputs:
            doc2 = request.inputs["document2"]
            format_kwargs["document2"] = doc2.get("content", "") if isinstance(doc2, dict) else str(doc2)

        # Format the prompt
        formatted_prompt = skill.system_prompt.format(
            **{k: v for k, v in format_kwargs.items() if f"{{{k}}}" in skill.system_prompt}
        )
    except KeyError as e:
        execution.status = SkillExecutionStatus.FAILED.value
        execution.error_message = f"Missing required input: {e}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required input: {e}"
        )

    # Get LLM and execute
    try:
        llm = LLMFactory.get_chat_model(
            provider=provider_type,
            model=model_name,
            temperature=0.7,
            max_tokens=4096,
        )

        response = await llm.ainvoke(formatted_prompt)
        output = response.content

        # Try to parse JSON responses
        if output.strip().startswith("{") or output.strip().startswith("["):
            try:
                import json
                output = json.loads(output)
            except json.JSONDecodeError:
                pass  # Keep as string

        execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        # Update execution record
        execution.status = SkillExecutionStatus.COMPLETED.value
        execution.output = {"result": output} if not isinstance(output, dict) else output
        execution.execution_time_ms = execution_time
        execution.model_used = model_name
        execution.provider_used = provider_type

        # Update skill stats
        skill.use_count += 1
        skill.last_used_at = datetime.utcnow()
        if skill.avg_execution_time_ms:
            skill.avg_execution_time_ms = (skill.avg_execution_time_ms + execution_time) // 2
        else:
            skill.avg_execution_time_ms = execution_time

        await db.commit()

        logger.info(
            "Skill executed successfully",
            skill_id=str(skill.id),
            execution_time_ms=execution_time,
            model=model_name,
        )

        return SkillExecuteResponse(
            output=output,
            execution_time_ms=execution_time,
            model_used=f"{provider_type}/{model_name}",
            tokens_used=None,
            execution_id=str(execution.id),
        )

    except Exception as e:
        execution.status = SkillExecutionStatus.FAILED.value
        execution.error_message = str(e)
        await db.commit()

        logger.error(
            "Skill execution failed",
            skill_id=str(skill.id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Skill execution failed"
        )


@router.post("", response_model=SkillCreateResponse)
async def create_skill(
    request: SkillCreateRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create a custom skill definition.

    Custom skills are stored per-user and can be executed like built-in skills.
    """
    # Generate skill_key if not provided
    skill_key = request.id or f"custom-{uuid4().hex[:8]}"
    user_uuid = _to_uuid(user.user_id)

    # Check if skill_key already exists for this user
    result = await db.execute(
        select(Skill).where(
            Skill.skill_key == skill_key,
            Skill.user_id == user_uuid,
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Skill with key '{skill_key}' already exists"
        )

    logger.info(
        "Creating custom skill",
        skill_key=skill_key,
        user_id=user.user_id,
    )

    # Create skill
    skill = Skill(
        skill_key=skill_key,
        name=request.name,
        description=request.description,
        category=request.category,
        icon=request.icon,
        tags=request.tags,
        system_prompt=request.system_prompt,
        inputs=[i.model_dump() for i in request.inputs],
        outputs=[o.model_dump() for o in request.outputs],
        user_id=user_uuid,
        is_public=request.is_public,
        is_builtin=False,
        is_active=True,
    )
    db.add(skill)
    await db.commit()
    await db.refresh(skill)

    return SkillCreateResponse(
        id=str(skill.id),
        skill_key=skill.skill_key,
        name=skill.name,
        message=f"Skill '{skill.name}' created successfully",
    )


@router.post("/import", response_model=SkillCreateResponse)
async def import_skill(
    request: SkillImportRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Import a skill from a JSON definition.
    """
    skill_data = request.skill_data

    if "name" not in skill_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skill definition must include 'name' field"
        )

    skill_key = skill_data.get("id", f"imported-{uuid4().hex[:8]}")
    user_uuid = _to_uuid(user.user_id)

    # Check if skill_key already exists for this user
    result = await db.execute(
        select(Skill).where(
            Skill.skill_key == skill_key,
            Skill.user_id == user_uuid,
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Skill with key '{skill_key}' already exists"
        )

    logger.info(
        "Importing skill",
        skill_key=skill_key,
        user_id=user.user_id,
    )

    # Create skill from imported data
    skill = Skill(
        skill_key=skill_key,
        name=skill_data["name"],
        description=skill_data.get("description", ""),
        category=skill_data.get("category", "custom"),
        icon=skill_data.get("icon", "zap"),
        tags=skill_data.get("tags", []),
        system_prompt=skill_data.get("system_prompt", "Process the input and provide a response.\n\nInput: {content}"),
        inputs=skill_data.get("inputs", []),
        outputs=skill_data.get("outputs", []),
        user_id=user_uuid,
        is_public=skill_data.get("is_public", False),
        is_builtin=False,
        is_active=True,
    )
    db.add(skill)
    await db.commit()
    await db.refresh(skill)

    return SkillCreateResponse(
        id=str(skill.id),
        skill_key=skill.skill_key,
        name=skill.name,
        message=f"Skill '{skill.name}' imported successfully",
    )


@router.get("/list")
async def list_skills(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    include_public: bool = Query(True, description="Include public skills"),
    builtin_only: bool = Query(False, description="Only show built-in skills"),
):
    """
    List all available skills (built-in, public, and user's custom skills).
    """
    try:
        # Ensure built-in skills exist
        await ensure_builtin_skills(db)
    except Exception as e:
        logger.warning("Failed to ensure builtin skills, continuing anyway", error=str(e))
        # Don't fail the whole request if builtin skills can't be created
        await db.rollback()

    try:
        user_uuid = _to_uuid(user.user_id)

        # Build query
        conditions = [Skill.is_active == True]

        # Access control: user's own skills, public skills, or built-in skills
        access_conditions = [Skill.user_id == user_uuid]
        if include_public:
            access_conditions.append(Skill.is_public == True)
        access_conditions.append(Skill.is_builtin == True)
        conditions.append(or_(*access_conditions))

        if category:
            conditions.append(Skill.category == category)

        if builtin_only:
            conditions.append(Skill.is_builtin == True)

        if search:
            safe = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            search_pattern = f"%{safe}%"
            conditions.append(
                or_(
                    Skill.name.ilike(search_pattern),
                    Skill.description.ilike(search_pattern),
                )
            )

        query = select(Skill).where(and_(*conditions)).order_by(Skill.is_builtin.desc(), Skill.use_count.desc())
        result = await db.execute(query)
        skills = result.scalars().all()

        # Format response
        skills_list = []
        for skill in skills:
            skills_list.append({
                "id": str(skill.id),
                "skill_key": skill.skill_key,
                "name": skill.name,
                "description": skill.description,
                "category": skill.category,
                "icon": skill.icon,
                "tags": skill.tags or [],
                "inputs": skill.inputs or [],
                "outputs": skill.outputs or [],
                "is_public": skill.is_public,
                "is_builtin": skill.is_builtin,
                "version": skill.version,
                "use_count": skill.use_count,
                "avg_execution_time_ms": skill.avg_execution_time_ms,
                "is_owner": skill.user_id == user_uuid if skill.user_id else False,
                "created_at": skill.created_at.isoformat() if skill.created_at else None,
            })

        return {"skills": skills_list, "total": len(skills_list)}

    except Exception as e:
        logger.error("Failed to list skills", error=str(e), user_id=user.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load skills"
        )


@router.get("/{skill_id}")
async def get_skill(
    skill_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get a specific skill by ID or skill_key.
    """
    await ensure_builtin_skills(db)
    skill = await get_skill_by_id_or_key(db, skill_id, user.user_id)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {skill_id}"
        )

    user_uuid = _to_uuid(user.user_id)
    return {
        "id": str(skill.id),
        "skill_key": skill.skill_key,
        "name": skill.name,
        "description": skill.description,
        "category": skill.category,
        "icon": skill.icon,
        "tags": skill.tags or [],
        "inputs": skill.inputs or [],
        "outputs": skill.outputs or [],
        "system_prompt": skill.system_prompt if skill.user_id == user_uuid else None,  # Only show prompt to owner
        "is_public": skill.is_public,
        "is_builtin": skill.is_builtin,
        "version": skill.version,
        "use_count": skill.use_count,
        "avg_execution_time_ms": skill.avg_execution_time_ms,
        "is_owner": skill.user_id == user_uuid if skill.user_id else False,
        "created_at": skill.created_at.isoformat() if skill.created_at else None,
        "updated_at": skill.updated_at.isoformat() if skill.updated_at else None,
    }


@router.put("/{skill_id}")
async def update_skill(
    skill_id: str,
    request: SkillUpdateRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update a custom skill. Only the owner can update their skills.
    """
    skill = await get_skill_by_id_or_key(db, skill_id, user.user_id)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {skill_id}"
        )

    if skill.is_builtin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot modify built-in skills"
        )

    user_uuid = _to_uuid(user.user_id)
    if skill.user_id != user_uuid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own skills"
        )

    # Update fields
    if request.name is not None:
        skill.name = request.name
    if request.description is not None:
        skill.description = request.description
    if request.category is not None:
        skill.category = request.category
    if request.system_prompt is not None:
        skill.system_prompt = request.system_prompt
    if request.inputs is not None:
        skill.inputs = [i.model_dump() for i in request.inputs]
    if request.outputs is not None:
        skill.outputs = [o.model_dump() for o in request.outputs]
    if request.tags is not None:
        skill.tags = request.tags
    if request.icon is not None:
        skill.icon = request.icon
    if request.is_public is not None:
        skill.is_public = request.is_public
    if request.is_active is not None:
        skill.is_active = request.is_active

    await db.commit()
    await db.refresh(skill)

    return {
        "id": str(skill.id),
        "skill_key": skill.skill_key,
        "name": skill.name,
        "message": f"Skill '{skill.name}' updated successfully",
    }


@router.delete("/{skill_id}")
async def delete_skill(
    skill_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete a custom skill. Only the owner can delete their skills.
    """
    skill = await get_skill_by_id_or_key(db, skill_id, user.user_id)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {skill_id}"
        )

    if skill.is_builtin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete built-in skills"
        )

    user_uuid = _to_uuid(user.user_id)
    if skill.user_id != user_uuid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own skills"
        )

    skill_name = skill.name
    await db.delete(skill)
    await db.commit()

    return {"message": f"Skill '{skill_name}' deleted successfully"}


@router.get("/{skill_id}/executions")
async def get_skill_executions(
    skill_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    Get execution history for a skill.
    """
    skill = await get_skill_by_id_or_key(db, skill_id, user.user_id)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {skill_id}"
        )

    user_uuid = _to_uuid(user.user_id)

    # Skill owner/builtin: show all executions; non-owner: only their own
    if skill.user_id == user_uuid or skill.is_builtin:
        query = select(SkillExecution).where(
            SkillExecution.skill_id == skill.id,
        ).order_by(SkillExecution.created_at.desc()).offset(offset).limit(limit)
        count_conditions = [SkillExecution.skill_id == skill.id]
    else:
        query = select(SkillExecution).where(
            SkillExecution.skill_id == skill.id,
            SkillExecution.user_id == user_uuid,
        ).order_by(SkillExecution.created_at.desc()).offset(offset).limit(limit)
        count_conditions = [SkillExecution.skill_id == skill.id, SkillExecution.user_id == user_uuid]

    result = await db.execute(query)
    executions = result.scalars().all()

    # Get total count
    count_query = select(func.count(SkillExecution.id)).where(
        *count_conditions,
    )
    total = (await db.execute(count_query)).scalar() or 0

    return {
        "executions": [
            {
                "id": str(e.id),
                "status": e.status,
                "execution_time_ms": e.execution_time_ms,
                "model_used": e.model_used,
                "provider_used": e.provider_used,
                "tokens_used": e.tokens_used,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "has_error": bool(e.error_message),
            }
            for e in executions
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# =============================================================================
# External Agent Endpoints
# =============================================================================

import httpx
import time
import json as json_module

from backend.services.encryption import encrypt_value, decrypt_value

# Sensitive fields in ExternalAgentAuth that must be encrypted at rest
_SENSITIVE_AUTH_FIELDS = ("api_key", "bearer_token", "password", "oauth2_client_secret")


def _encrypt_auth_data(auth_data: dict) -> dict:
    """Encrypt sensitive fields in external agent auth config before storage."""
    for field in _SENSITIVE_AUTH_FIELDS:
        value = auth_data.get(field)
        if value:
            auth_data[field] = encrypt_value(value)
    return auth_data


def _decrypt_auth_data(auth_data: dict) -> dict:
    """Decrypt sensitive fields in external agent auth config for use."""
    for field in _SENSITIVE_AUTH_FIELDS:
        value = auth_data.get(field)
        if value:
            try:
                auth_data[field] = decrypt_value(value)
            except ValueError:
                pass  # Already plaintext (pre-encryption records)
    return auth_data


def _apply_json_path(data: Any, path: str) -> Any:
    """Apply a JSON path to extract nested values (e.g., 'result.data.text')."""
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            idx = int(part)
            current = current[idx] if idx < len(current) else None
        else:
            return None
        if current is None:
            return None
    return current


def _set_json_path(data: Dict, path: str, value: Any) -> Dict:
    """Set a value at a JSON path (e.g., 'data.prompt')."""
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return data


def _build_auth_headers(auth: ExternalAgentAuth) -> Dict[str, str]:
    """Build authentication headers from auth config."""
    headers = {}

    if auth.type == "none":
        pass
    elif auth.type == "api_key":
        if auth.api_key:
            header_name = auth.api_key_header or "X-API-Key"
            headers[header_name] = auth.api_key
    elif auth.type == "bearer":
        if auth.bearer_token:
            headers["Authorization"] = f"Bearer {auth.bearer_token}"
    elif auth.type == "basic":
        if auth.username and auth.password:
            import base64
            credentials = base64.b64encode(f"{auth.username}:{auth.password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
    elif auth.type == "custom":
        if auth.custom_headers:
            headers.update(auth.custom_headers)

    return headers


def _is_safe_url(url: str) -> bool:
    """Validate URL is not targeting internal/private services (SSRF prevention)."""
    from urllib.parse import urlparse
    import ipaddress

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        # Block known internal/metadata hostnames
        blocked = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254", "[::1]"}
        if hostname in blocked:
            return False
        # Block private and link-local IP ranges
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except ValueError:
            # Hostname, not IP — resolve DNS and check the resolved IP
            import socket
            try:
                resolved_ip = ipaddress.ip_address(socket.gethostbyname(hostname))
                if resolved_ip.is_private or resolved_ip.is_loopback or resolved_ip.is_link_local or resolved_ip.is_reserved:
                    return False
            except socket.gaierror:
                return False  # DNS resolution failed — deny unverifiable addresses
        return True
    except Exception:
        return False


@router.post("/external/test", response_model=ExternalAgentTestResponse)
async def test_external_agent(
    request: ExternalAgentTestRequest,
    user: AuthenticatedUser,
):
    """
    Test an external agent configuration before creating the skill.
    Verifies connectivity, authentication, and response format.
    """
    # SSRF prevention: validate endpoint URL
    if not _is_safe_url(request.endpoint_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid endpoint URL: private/internal addresses are not allowed",
        )

    start_time = time.time()

    try:
        headers = _build_auth_headers(request.auth)
        headers["Content-Type"] = "application/json"

        async with httpx.AsyncClient(timeout=request.timeout_seconds) as client:
            if request.method.upper() == "GET":
                response = await client.get(request.endpoint_url, headers=headers)
            elif request.method.upper() == "POST":
                response = await client.post(
                    request.endpoint_url,
                    headers=headers,
                    json=request.request_body,
                )
            elif request.method.upper() == "PUT":
                response = await client.put(
                    request.endpoint_url,
                    headers=headers,
                    json=request.request_body,
                )
            else:
                return ExternalAgentTestResponse(
                    success=False,
                    status_code=None,
                    response_body=None,
                    error=f"Unsupported HTTP method: {request.method}",
                    latency_ms=int((time.time() - start_time) * 1000),
                )

            latency_ms = int((time.time() - start_time) * 1000)

            # Try to parse response body
            response_body = None
            try:
                response_body = response.json()
            except Exception:
                response_body = response.text[:1000] if response.text else None

            success = 200 <= response.status_code < 300

            return ExternalAgentTestResponse(
                success=success,
                status_code=response.status_code,
                response_body=response_body,
                error=None if success else f"HTTP {response.status_code}",
                latency_ms=latency_ms,
            )

    except httpx.TimeoutException:
        return ExternalAgentTestResponse(
            success=False,
            status_code=None,
            response_body=None,
            error=f"Request timed out after {request.timeout_seconds} seconds",
            latency_ms=int((time.time() - start_time) * 1000),
        )
    except httpx.ConnectError as e:
        return ExternalAgentTestResponse(
            success=False,
            status_code=None,
            response_body=None,
            error=f"Connection failed: {str(e)}",
            latency_ms=int((time.time() - start_time) * 1000),
        )
    except Exception as e:
        return ExternalAgentTestResponse(
            success=False,
            status_code=None,
            response_body=None,
            error=f"Error: {str(e)}",
            latency_ms=int((time.time() - start_time) * 1000),
        )


@router.post("/external", response_model=SkillCreateResponse)
async def create_external_agent_skill(
    request: ExternalAgentCreateRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create a skill from an external agent/API.
    This wraps an external API endpoint as a skill that users can execute.
    """
    user_uuid = _to_uuid(user.user_id)

    # Generate skill key
    skill_key = f"external-{request.name.lower().replace(' ', '-')}-{str(uuid4())[:8]}"

    # Store the external agent configuration in the system prompt (as JSON metadata)
    external_config = {
        "type": "external_agent",
        "endpoint_url": request.endpoint_url,
        "method": request.method,
        "content_type": request.content_type,
        "auth": _encrypt_auth_data(request.auth.model_dump()),
        "input_mapping": [m.model_dump() for m in request.input_mapping],
        "output_mapping": [m.model_dump() for m in request.output_mapping],
        "request_template": request.request_template,
        "timeout_seconds": request.timeout_seconds,
        "retry_count": request.retry_count,
        "rate_limit_per_minute": request.rate_limit_per_minute,
    }

    # Create skill with external config stored as system_prompt JSON
    skill = Skill(
        skill_key=skill_key,
        name=request.name,
        description=request.description,
        category=request.category,
        icon=request.icon,
        tags=request.tags,
        system_prompt=f"__EXTERNAL_AGENT_CONFIG__:{json_module.dumps(external_config)}",
        inputs=[inp.model_dump() for inp in request.inputs],
        outputs=[out.model_dump() for out in request.outputs],
        is_builtin=False,
        is_public=request.is_public,
        is_active=True,
        user_id=user_uuid,
        version="1.0.0",
    )

    db.add(skill)
    await db.commit()
    await db.refresh(skill)

    logger.info(
        "External agent skill created",
        skill_id=str(skill.id),
        skill_key=skill_key,
        endpoint_url=request.endpoint_url,
    )

    return SkillCreateResponse(
        id=str(skill.id),
        skill_key=skill_key,
        name=request.name,
        message=f"External agent skill '{request.name}' created successfully",
    )


@router.post("/external/{skill_id}/execute", response_model=SkillExecuteResponse)
async def execute_external_agent_skill(
    skill_id: str,
    inputs: Dict[str, Any],
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Execute an external agent skill by calling the configured API endpoint.
    """
    start_time = time.time()

    # Get the skill
    skill = await get_skill_by_id_or_key(db, skill_id, user.user_id)
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {skill_id}",
        )

    # Check if it's an external agent skill
    if not skill.system_prompt.startswith("__EXTERNAL_AGENT_CONFIG__:"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This endpoint is only for external agent skills",
        )

    # Parse the external config
    try:
        config_json = skill.system_prompt.replace("__EXTERNAL_AGENT_CONFIG__:", "")
        config = json_module.loads(config_json)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse external agent configuration",
        )

    # SSRF prevention: validate endpoint URL from stored config
    endpoint_url = config.get("endpoint_url", "")
    if not _is_safe_url(endpoint_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid endpoint URL: private/internal addresses are not allowed",
        )

    user_uuid = _to_uuid(user.user_id)

    # Build the request body from inputs
    request_body = config.get("request_template", {}).copy() if config.get("request_template") else {}

    for mapping in config.get("input_mapping", []):
        skill_input = mapping["skill_input"]
        api_path = mapping["api_path"]
        transform = mapping.get("transform")

        value = inputs.get(skill_input)
        if value is not None:
            if transform == "stringify":
                value = str(value)
            elif transform == "parse_json":
                try:
                    value = json_module.loads(value) if isinstance(value, str) else value
                except Exception:
                    pass
            _set_json_path(request_body, api_path, value)

    # Build auth headers (decrypt credentials from storage)
    auth_data = _decrypt_auth_data(config.get("auth", {}))
    auth_config = ExternalAgentAuth(**auth_data)
    headers = _build_auth_headers(auth_config)
    headers["Content-Type"] = config.get("content_type", "application/json")

    # Execute the request
    execution = SkillExecution(
        skill_id=skill.id,
        user_id=user_uuid,
        inputs=inputs,
        status=SkillExecutionStatus.RUNNING,
        model_used="external_api",
        provider_used="external",
    )
    db.add(execution)
    await db.commit()

    try:
        timeout = min(config.get("timeout_seconds", 60), 120)  # Max 2 minutes per request
        retry_count = min(config.get("retry_count", 1), 5)  # Max 5 retries

        last_error = None
        response = None

        for attempt in range(retry_count):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    method = config.get("method", "POST").upper()
                    endpoint_url = config.get("endpoint_url")

                    if method == "GET":
                        response = await client.get(endpoint_url, headers=headers)
                    elif method == "POST":
                        response = await client.post(endpoint_url, headers=headers, json=request_body)
                    elif method == "PUT":
                        response = await client.put(endpoint_url, headers=headers, json=request_body)
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unsupported HTTP method: {method}",
                        )

                    if 200 <= response.status_code < 300:
                        break
                    else:
                        last_error = f"HTTP {response.status_code}: {response.text[:500]}"

            except httpx.TimeoutException:
                last_error = f"Request timed out after {timeout} seconds"
            except httpx.ConnectError as e:
                last_error = f"Connection failed: {str(e)}"
            except Exception as e:
                last_error = str(e)

        if response is None or not (200 <= response.status_code < 300):
            execution.status = SkillExecutionStatus.FAILED
            execution.error_message = last_error
            execution.execution_time_ms = int((time.time() - start_time) * 1000)
            await db.commit()

            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"External agent request failed: {last_error}",
            )

        # Parse response
        try:
            response_data = response.json()
        except Exception:
            response_data = {"raw": response.text}

        # Apply output mapping
        output = {}
        for mapping in config.get("output_mapping", []):
            api_path = mapping["api_path"]
            skill_output = mapping["skill_output"]
            transform = mapping.get("transform")

            value = _apply_json_path(response_data, api_path)
            if transform == "stringify":
                value = str(value) if value is not None else None
            elif transform == "parse_json":
                try:
                    value = json_module.loads(value) if isinstance(value, str) else value
                except Exception:
                    pass
            output[skill_output] = value

        # If no output mapping, use raw response
        if not output:
            output = response_data

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Update execution record
        execution.status = SkillExecutionStatus.COMPLETED
        execution.output = output
        execution.execution_time_ms = execution_time_ms
        await db.commit()

        # Update skill use count
        skill.use_count = (skill.use_count or 0) + 1
        await db.commit()

        return SkillExecuteResponse(
            output=output,
            execution_time_ms=execution_time_ms,
            model_used="external_api",
            tokens_used=None,
            execution_id=str(execution.id),
        )

    except HTTPException:
        raise
    except Exception as e:
        execution.status = SkillExecutionStatus.FAILED
        execution.error_message = str(e)
        execution.execution_time_ms = int((time.time() - start_time) * 1000)
        await db.commit()

        logger.error("External agent execution failed", error=str(e), skill_id=skill_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="External agent execution failed",
        )


@router.get("/external/{skill_id}/config")
async def get_external_agent_config(
    skill_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get the configuration for an external agent skill (without secrets).
    """
    skill = await get_skill_by_id_or_key(db, skill_id, user.user_id)
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {skill_id}",
        )

    # Check ownership
    user_uuid = _to_uuid(user.user_id)
    if skill.user_id != user_uuid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view config for your own skills",
        )

    if not skill.system_prompt.startswith("__EXTERNAL_AGENT_CONFIG__:"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This skill is not an external agent",
        )

    try:
        config_json = skill.system_prompt.replace("__EXTERNAL_AGENT_CONFIG__:", "")
        config = json_module.loads(config_json)

        # Remove secrets from auth
        if "auth" in config:
            auth = config["auth"]
            if auth.get("api_key"):
                auth["api_key"] = "***hidden***"
            if auth.get("bearer_token"):
                auth["bearer_token"] = "***hidden***"
            if auth.get("password"):
                auth["password"] = "***hidden***"
            if auth.get("oauth2_client_secret"):
                auth["oauth2_client_secret"] = "***hidden***"

        return {
            "skill_id": str(skill.id),
            "skill_key": skill.skill_key,
            "name": skill.name,
            "config": config,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse configuration",
        )


# =============================================================================
# Publishing Endpoints
# =============================================================================

class SkillPublishRequest(BaseModel):
    """Request to publish a skill."""
    rate_limit: int = Field(default=100, description="Max executions per minute")
    allowed_domains: List[str] = Field(default=["*"], description="Allowed origin domains")
    require_api_key: bool = Field(default=False, description="Require API key for access")
    custom_slug: Optional[str] = Field(None, description="Custom URL slug")
    branding: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom branding")


class SkillPublishResponse(BaseModel):
    """Response from publishing a skill."""
    skill_id: str
    public_slug: str
    public_url: str
    embed_code: str
    is_published: bool


def _generate_slug(name: str) -> str:
    """Generate a URL-friendly slug from name."""
    import re
    slug = name.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return f"{slug}-{uuid4().hex[:8]}"


@router.post("/{skill_id}/publish", response_model=SkillPublishResponse)
async def publish_skill(
    skill_id: str,
    request: SkillPublishRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Publish a skill for external access via public URL.

    Once published, the skill can be accessed without authentication
    at /api/v1/public/skills/{public_slug}.
    """
    skill = await get_skill_by_id_or_key(db, skill_id, user.user_id)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {skill_id}",
        )

    user_uuid = _to_uuid(user.user_id)
    if skill.user_id != user_uuid and not skill.is_builtin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only publish your own skills",
        )

    # Generate or use custom slug
    if request.custom_slug:
        # Check if slug is already taken
        existing = await db.execute(
            select(Skill).where(
                Skill.public_slug == request.custom_slug,
                Skill.id != skill.id,
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Slug '{request.custom_slug}' is already taken",
            )
        public_slug = request.custom_slug
    else:
        public_slug = _generate_slug(skill.name)

    # Update skill
    skill.is_published = True
    skill.public_slug = public_slug
    skill.publish_config = {
        "rate_limit": request.rate_limit,
        "allowed_domains": request.allowed_domains,
        "require_api_key": request.require_api_key,
        "branding": request.branding,
    }

    await db.commit()
    await db.refresh(skill)

    # Generate URLs
    base_url = settings.server.frontend_url or "http://localhost:3000"
    public_url = f"{base_url}/s/{public_slug}"
    api_url = f"{settings.server.backend_url or 'http://localhost:8000'}/api/v1/public/skills/{public_slug}"

    # Generate embed code
    embed_code = f'''<iframe
  src="{public_url}/embed"
  width="400"
  height="600"
  frameborder="0"
  allow="clipboard-write"
></iframe>'''

    logger.info(
        "Skill published",
        skill_id=str(skill.id),
        public_slug=public_slug,
        user_id=user.user_id,
    )

    return SkillPublishResponse(
        skill_id=str(skill.id),
        public_slug=public_slug,
        public_url=public_url,
        embed_code=embed_code,
        is_published=True,
    )


@router.post("/{skill_id}/unpublish")
async def unpublish_skill(
    skill_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Unpublish a skill, removing public access."""
    skill = await get_skill_by_id_or_key(db, skill_id, user.user_id)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {skill_id}",
        )

    user_uuid = _to_uuid(user.user_id)
    if skill.user_id != user_uuid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only unpublish your own skills",
        )

    skill.is_published = False
    # Keep the slug for re-publishing

    await db.commit()

    return {"message": f"Skill '{skill.name}' unpublished successfully"}


@router.get("/{skill_id}/publish-status")
async def get_skill_publish_status(
    skill_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Get the publish status and public URL for a skill."""
    skill = await get_skill_by_id_or_key(db, skill_id, user.user_id)

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {skill_id}",
        )

    base_url = settings.server.frontend_url or "http://localhost:3000"

    return {
        "skill_id": str(skill.id),
        "is_published": skill.is_published,
        "public_slug": skill.public_slug,
        "public_url": f"{base_url}/s/{skill.public_slug}" if skill.public_slug else None,
        "publish_config": skill.publish_config,
    }
