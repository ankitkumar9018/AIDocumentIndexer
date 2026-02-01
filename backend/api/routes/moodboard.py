"""
AIDocumentIndexer - Mood Board API Routes
==========================================

Endpoints for AI-powered mood board generation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
import json

from uuid import UUID
from sqlalchemy import select, func, and_

from backend.services.llm import LLMFactory
from backend.services.llm_provider import LLMProviderService
from backend.core.config import settings
from backend.db.database import get_async_session
from backend.db.models import MoodBoard, MoodBoardStatus
from backend.api.middleware.auth import AuthenticatedUser

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class MoodBoardGenerateRequest(BaseModel):
    """Request to generate a mood board."""
    name: str = Field(..., description="Mood board name")
    description: Optional[str] = Field(None, description="Description or concept")
    colors: List[str] = Field(default_factory=list, description="Color palette (hex codes)")
    keywords: List[str] = Field(default_factory=list, description="Style keywords")
    style_notes: Optional[str] = Field(None, description="Additional style notes")
    provider_id: Optional[str] = Field(None, description="LLM provider ID to use")
    model: Optional[str] = Field(None, description="Model name to use")


class GeneratedSuggestions(BaseModel):
    """AI-generated suggestions for the mood board."""
    typography: List[str] = Field(default_factory=list, description="Suggested fonts")
    additional_colors: List[str] = Field(default_factory=list, description="Complementary colors")
    mood_keywords: List[str] = Field(default_factory=list, description="Additional mood keywords")
    inspiration_notes: str = Field(..., description="AI-generated inspiration notes")
    design_direction: Optional[str] = Field(None, description="Design direction suggestions")
    color_psychology: Optional[Dict[str, str]] = Field(None, description="Color psychology insights")


class MoodBoardGenerateResponse(BaseModel):
    """Response from mood board generation."""
    id: str = Field(..., description="Generated mood board ID")
    name: str
    description: Optional[str]
    colors: List[str]
    keywords: List[str]
    style_notes: Optional[str]
    generated_suggestions: GeneratedSuggestions
    created_at: str
    model_used: str


class MoodBoardSaveRequest(BaseModel):
    """Request to save a mood board."""
    mood_board: Dict[str, Any] = Field(..., description="Mood board data to save")


# =============================================================================
# Mood Board Generation Prompt
# =============================================================================

MOODBOARD_PROMPT = """You are an expert visual designer and brand strategist. Based on the following mood board inputs, generate creative suggestions to enhance the visual direction.

**Mood Board Name:** {name}
**Description/Concept:** {description}
**Color Palette:** {colors}
**Style Keywords:** {keywords}
**Additional Notes:** {style_notes}

Analyze the inputs and provide:

1. **Typography Suggestions**: 3 font families that complement the mood (mix of serif, sans-serif, display fonts as appropriate)

2. **Complementary Colors**: 3 additional hex colors that enhance the existing palette

3. **Mood Keywords**: 3-5 additional mood/style descriptors that align with the concept

4. **Inspiration Notes**: A 2-3 sentence description of the overall aesthetic direction, including what emotions it evokes and what industries/use cases it would suit

5. **Design Direction**: Specific guidance on visual elements like shapes, textures, imagery styles

6. **Color Psychology**: Brief insight into what each main color communicates

Respond ONLY with a valid JSON object in this exact format:
{{
  "typography": ["Font1", "Font2", "Font3"],
  "additional_colors": ["#hexcode1", "#hexcode2", "#hexcode3"],
  "mood_keywords": ["keyword1", "keyword2", "keyword3"],
  "inspiration_notes": "Your 2-3 sentence aesthetic description",
  "design_direction": "Specific guidance on visual elements",
  "color_psychology": {{
    "#hexcode": "what this color communicates"
  }}
}}"""


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/generate", response_model=MoodBoardGenerateResponse)
async def generate_mood_board(
    request: MoodBoardGenerateRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Generate AI-powered suggestions for a mood board.

    Uses the configured LLM to analyze colors, keywords, and concept
    to provide typography, color, and design direction suggestions.
    """
    logger.info(
        "Generating mood board",
        name=request.name,
        user_id=user.user_id,
        colors_count=len(request.colors),
        keywords_count=len(request.keywords),
    )

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

    # Format the prompt
    colors_str = ", ".join(request.colors) if request.colors else "Not specified"
    keywords_str = ", ".join(request.keywords) if request.keywords else "Not specified"

    formatted_prompt = MOODBOARD_PROMPT.format(
        name=request.name,
        description=request.description or "Not specified",
        colors=colors_str,
        keywords=keywords_str,
        style_notes=request.style_notes or "None provided",
    )

    # Get LLM and execute
    try:
        llm = LLMFactory.get_chat_model(
            provider=provider_type,
            model=model_name,
            temperature=0.8,  # Slightly higher for creativity
            max_tokens=2048,
        )

        response = await llm.ainvoke(formatted_prompt)
        output = response.content

        # Parse JSON response
        try:
            # Clean up the response if needed
            output_clean = output.strip()
            if output_clean.startswith("```json"):
                output_clean = output_clean[7:]
            if output_clean.startswith("```"):
                output_clean = output_clean[3:]
            if output_clean.endswith("```"):
                output_clean = output_clean[:-3]

            suggestions_data = json.loads(output_clean.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Provide fallback suggestions
            suggestions_data = {
                "typography": ["Inter", "Playfair Display", "Space Grotesk"],
                "additional_colors": ["#f97316", "#22c55e", "#0ea5e9"],
                "mood_keywords": ["modern", "elegant", "sophisticated"],
                "inspiration_notes": f"Based on the {request.name} concept with {keywords_str} aesthetic, this mood board suggests a contemporary direction with the provided color palette.",
                "design_direction": "Focus on clean lines and balanced compositions.",
                "color_psychology": {},
            }

        # Build response
        suggestions = GeneratedSuggestions(
            typography=suggestions_data.get("typography", ["Inter", "Roboto", "Open Sans"]),
            additional_colors=suggestions_data.get("additional_colors", ["#f97316", "#22c55e", "#0ea5e9"]),
            mood_keywords=suggestions_data.get("mood_keywords", ["modern", "clean", "professional"]),
            inspiration_notes=suggestions_data.get("inspiration_notes", "A sophisticated visual direction."),
            design_direction=suggestions_data.get("design_direction"),
            color_psychology=suggestions_data.get("color_psychology"),
        )

        board_id = str(uuid4())

        logger.info(
            "Mood board generated successfully",
            board_id=board_id,
            model=model_name,
        )

        return MoodBoardGenerateResponse(
            id=board_id,
            name=request.name,
            description=request.description,
            colors=request.colors,
            keywords=request.keywords,
            style_notes=request.style_notes,
            generated_suggestions=suggestions,
            created_at=datetime.utcnow().isoformat(),
            model_used=f"{provider_type}/{model_name}",
        )

    except Exception as e:
        logger.error(
            "Mood board generation failed",
            name=request.name,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mood board generation failed: {str(e)}"
        )


@router.post("/save")
async def save_mood_board(
    request: MoodBoardSaveRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Save a generated mood board to the database for later use.
    """
    board_data = request.mood_board
    logger.info(
        "Saving mood board",
        board_name=board_data.get("name"),
        user_id=user.user_id,
    )

    # Extract fields from the mood board data
    name = board_data.get("name", "Untitled Mood Board")
    description = board_data.get("description")
    prompt = board_data.get("style_notes") or json.dumps({
        "colors": board_data.get("colors", []),
        "keywords": board_data.get("keywords", []),
    })

    # Extract generated suggestions if present
    suggestions = board_data.get("generated_suggestions", {})
    color_palette = board_data.get("colors", [])
    if suggestions.get("additional_colors"):
        color_palette = color_palette + suggestions.get("additional_colors", [])

    style_tags = board_data.get("keywords", [])
    if suggestions.get("mood_keywords"):
        style_tags = style_tags + suggestions.get("mood_keywords", [])

    # Create the MoodBoard record
    mood_board = MoodBoard(
        user_id=user.user_id,
        name=name,
        description=description,
        prompt=prompt,
        status=MoodBoardStatus.COMPLETED.value,
        images=[],  # User can add images later
        themes=suggestions.get("typography", []),
        color_palette=color_palette,
        style_tags=style_tags,
        is_public=False,
    )

    db.add(mood_board)
    await db.commit()
    await db.refresh(mood_board)

    logger.info(
        "Mood board saved successfully",
        board_id=str(mood_board.id),
        user_id=user.user_id,
    )

    return {
        "id": str(mood_board.id),
        "message": "Mood board saved successfully",
        "saved_at": mood_board.created_at.isoformat(),
    }


@router.get("/list")
async def list_mood_boards(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    limit: int = 50,
    offset: int = 0,
):
    """
    List saved mood boards for the current user.
    """
    conditions = [MoodBoard.user_id == user.user_id]

    # Get total count
    count_query = select(func.count(MoodBoard.id)).where(and_(*conditions))
    total = (await db.execute(count_query)).scalar() or 0

    # Get mood boards
    query = (
        select(MoodBoard)
        .where(and_(*conditions))
        .order_by(MoodBoard.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(query)
    boards = result.scalars().all()

    return {
        "mood_boards": [
            {
                "id": str(b.id),
                "name": b.name,
                "description": b.description,
                "prompt": b.prompt,
                "status": b.status,
                "thumbnail_url": b.thumbnail_url,
                "style_tags": b.style_tags or [],
                "created_at": b.created_at.isoformat(),
            }
            for b in boards
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{board_id}")
async def get_mood_board(
    board_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get a specific mood board by ID.
    """
    try:
        board_uuid = UUID(board_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid mood board ID")

    result = await db.execute(
        select(MoodBoard).where(
            MoodBoard.id == board_uuid,
            MoodBoard.user_id == user.user_id,
        )
    )
    board = result.scalar_one_or_none()

    if not board:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mood board {board_id} not found"
        )

    return {
        "id": str(board.id),
        "name": board.name,
        "description": board.description,
        "prompt": board.prompt,
        "status": board.status,
        "images": board.images or [],
        "themes": board.themes or [],
        "color_palette": board.color_palette or [],
        "style_tags": board.style_tags or [],
        "thumbnail_url": board.thumbnail_url,
        "is_public": board.is_public,
        "created_at": board.created_at.isoformat(),
        "updated_at": board.updated_at.isoformat(),
    }


@router.delete("/{board_id}")
async def delete_mood_board(
    board_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete a mood board.
    """
    try:
        board_uuid = UUID(board_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid mood board ID")

    result = await db.execute(
        select(MoodBoard).where(
            MoodBoard.id == board_uuid,
            MoodBoard.user_id == user.user_id,
        )
    )
    board = result.scalar_one_or_none()

    if not board:
        raise HTTPException(status_code=404, detail="Mood board not found")

    logger.info(
        "Deleting mood board",
        board_id=board_id,
        user_id=user.user_id,
    )

    await db.delete(board)
    await db.commit()

    return {
        "message": f"Mood board {board_id} deleted",
        "deleted_at": datetime.utcnow().isoformat(),
    }
