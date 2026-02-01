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

from backend.services.llm import LLMFactory
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
    system_prompt: str = Field(..., description="System prompt for the skill")
    inputs: List[SkillInput] = Field(default_factory=list, description="Input schema")
    outputs: List[SkillOutput] = Field(default_factory=list, description="Output schema")
    tags: List[str] = Field(default_factory=list, description="Skill tags")
    icon: str = Field(default="zap", description="Icon name")
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

    # Get provider configuration
    provider_type = "openai"
    model_name = request.model or "gpt-4o"

    if request.provider_id:
        try:
            provider = await LLMProviderService.get_provider(db, request.provider_id)
            if provider:
                provider_type = provider.provider_type
                model_name = request.model or provider.default_chat_model or "gpt-4o"
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
            detail=f"Skill execution failed: {str(e)}"
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
    # Ensure built-in skills exist
    await ensure_builtin_skills(db)

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
        search_pattern = f"%{search}%"
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

    # Only show user's own executions unless they own the skill
    if skill.user_id == user_uuid or skill.is_builtin:
        # Show all executions by this user
        query = select(SkillExecution).where(
            SkillExecution.skill_id == skill.id,
            SkillExecution.user_id == user_uuid,
        ).order_by(SkillExecution.created_at.desc()).offset(offset).limit(limit)
    else:
        # Only show this user's executions
        query = select(SkillExecution).where(
            SkillExecution.skill_id == skill.id,
            SkillExecution.user_id == user_uuid,
        ).order_by(SkillExecution.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    executions = result.scalars().all()

    # Get total count
    count_query = select(func.count(SkillExecution.id)).where(
        SkillExecution.skill_id == skill.id,
        SkillExecution.user_id == user_uuid,
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
