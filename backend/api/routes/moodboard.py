"""
AIDocumentIndexer - Mood Board API Routes
==========================================

Endpoints for AI-powered mood board generation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
import json

from uuid import UUID
from sqlalchemy import select, func, and_

import asyncio

from backend.services.llm import LLMFactory, llm_config
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

class MultiLLMConfig(BaseModel):
    """Configuration for multi-LLM generation."""
    providers: List[Dict[str, str]] = Field(
        ...,
        description="List of {provider_id, model} pairs",
        min_length=1,
        max_length=3,
    )
    merge_strategy: str = Field(
        default="best_of",
        description="How to combine: 'best_of' or 'merge'"
    )


class MoodBoardGenerateRequest(BaseModel):
    """Request to generate a mood board."""
    name: str = Field(..., description="Mood board name")
    description: Optional[str] = Field(None, description="Description or concept")
    colors: List[str] = Field(default_factory=list, description="Color palette (hex codes)")
    keywords: List[str] = Field(default_factory=list, description="Style keywords")
    style_notes: Optional[str] = Field(None, description="Additional style notes")
    provider_id: Optional[str] = Field(None, description="LLM provider ID to use")
    model: Optional[str] = Field(None, description="Model name to use")
    # Document inspiration
    use_existing_docs: bool = Field(default=False, description="Use documents as inspiration")
    collection_filters: Optional[List[str]] = Field(None, description="Filter collections for doc inspiration")
    folder_id: Optional[str] = Field(None, description="Folder scope for doc inspiration")
    include_subfolders: bool = Field(default=False, description="Include subfolders")
    max_doc_sources: int = Field(default=5, ge=1, le=20, description="Max documents to use")
    # Dual mode
    dual_mode: bool = Field(default=False, description="Combine docs + general AI")
    # Multi-LLM
    multi_llm: Optional[MultiLLMConfig] = Field(None, description="Multi-LLM config")
    # Intelligence features (parity with chat/doc-gen pipeline)
    intelligence_level: Optional[str] = Field(
        default=None,
        pattern="^(basic|standard|enhanced|maximum)$",
        description="Intelligence level for source retrieval"
    )
    enhance_query: Optional[bool] = Field(None, description="Enable query enhancement for source search")
    skip_cache: bool = Field(default=False, description="Skip retrieval cache for fresh results")
    enable_cot: bool = Field(default=False, description="Enable chain-of-thought reasoning for generation")
    enable_verification: bool = Field(default=False, description="Verify generated content against source documents")
    temperature_override: Optional[float] = Field(
        default=None, ge=0.0, le=2.0,
        description="Override default temperature for generation"
    )


class GeneratedSuggestions(BaseModel):
    """AI-generated suggestions for the mood board."""
    typography: List[Any] = Field(default_factory=list, description="Fonts — strings or {font, role, rationale, sample_text}")
    additional_colors: List[Any] = Field(default_factory=list, description="Colors — strings or {hex, role, name}")
    mood_keywords: List[str] = Field(default_factory=list, description="Additional mood keywords")
    inspiration_notes: str = Field(default="", description="AI-generated inspiration notes")
    design_direction: Optional[str] = Field(None, description="Design direction suggestions")
    color_psychology: Optional[Dict[str, Any]] = Field(None, description="Color psychology insights")
    visual_narrative: Optional[str] = Field(None, description="Vivid creative brief")
    design_system: Optional[Dict[str, str]] = Field(None, description="Design system notes")
    anti_patterns: Optional[List[str]] = Field(None, description="What to avoid")
    image_search_terms: Optional[List[str]] = Field(None, description="Unsplash-style image search terms")


def _normalize_suggestions(raw: dict) -> dict:
    """Normalize the new rich prompt format into a backward-compatible structure."""
    normalized = dict(raw)

    # Typography: [{font, role, rationale}] → keep as-is but also provide flat list
    typo = raw.get("typography", [])
    if typo and isinstance(typo[0], dict):
        normalized["typography"] = typo
        normalized["_typography_flat"] = [t.get("font", t.get("name", "")) for t in typo]
    else:
        normalized["_typography_flat"] = typo

    # Colors: [{hex, role, name}] → keep as-is but also provide flat list
    colors = raw.get("additional_colors", [])
    if colors and isinstance(colors[0], dict):
        normalized["additional_colors"] = colors
        normalized["_colors_flat"] = [c.get("hex", c.get("color", "")) for c in colors]
    else:
        normalized["_colors_flat"] = colors

    # Color psychology: {"#hex": {"meaning":..., "pair_with":...}} → flatten for old format
    psych = raw.get("color_psychology", {})
    flat_psych = {}
    for hex_val, info in psych.items():
        if isinstance(info, dict):
            flat_psych[hex_val] = info.get("meaning", str(info))
        else:
            flat_psych[hex_val] = str(info)
    normalized["_color_psychology_flat"] = flat_psych

    # Visual narrative → inspiration_notes backward compat
    if raw.get("visual_narrative") and not raw.get("inspiration_notes"):
        normalized["inspiration_notes"] = raw["visual_narrative"]

    # Pass through image_search_terms
    if raw.get("image_search_terms"):
        normalized["image_search_terms"] = raw["image_search_terms"]

    return normalized


async def _call_single_llm(
    provider_type: str,
    model_name: Optional[str],
    prompt: str,
    temperature_override: Optional[float] = None,
) -> Optional[dict]:
    """Call a single LLM and parse its JSON response. Returns None on failure."""
    try:
        llm = LLMFactory.get_chat_model(
            provider=provider_type,
            model=model_name,
            temperature=temperature_override if temperature_override is not None else 0.8,
            max_tokens=2048,
        )
        response = await llm.ainvoke(prompt)
        output = response.content.strip()
        if output.startswith("```json"):
            output = output[7:]
        if output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]
        return json.loads(output.strip())
    except Exception as e:
        logger.warning("Multi-LLM call failed", provider=provider_type, model=model_name, error=str(e))
        return None


def _score_suggestion(data: dict) -> int:
    """Score how complete a suggestion dict is (more fields = higher)."""
    score = 0
    for key in ("typography", "additional_colors", "mood_keywords"):
        val = data.get(key, [])
        score += len(val) if isinstance(val, list) else 0
    for key in ("visual_narrative", "inspiration_notes", "design_direction"):
        val = data.get(key)
        score += len(val) if isinstance(val, str) and val else 0
    if data.get("color_psychology"):
        score += len(data["color_psychology"]) * 10
    if data.get("design_system"):
        score += len(data["design_system"]) * 5
    if data.get("anti_patterns"):
        score += len(data["anti_patterns"]) * 5
    return score


def _merge_suggestions(results: list[dict]) -> dict:
    """Merge multiple LLM suggestion dicts, picking the best parts from each."""
    if len(results) == 1:
        return results[0]

    merged: dict = {}

    # Typography: pick the set with richest rationale
    best_typo = max(results, key=lambda r: sum(
        len(t.get("rationale", "")) if isinstance(t, dict) else 0
        for t in r.get("typography", [])
    ))
    merged["typography"] = best_typo.get("typography", [])

    # Colors: pick the set with most named colors
    best_colors = max(results, key=lambda r: sum(
        1 for c in r.get("additional_colors", []) if isinstance(c, dict) and c.get("name")
    ))
    merged["additional_colors"] = best_colors.get("additional_colors", [])

    # Keywords: union from all, deduplicated
    all_kw: list[str] = []
    seen_kw: set[str] = set()
    for r in results:
        for kw in r.get("mood_keywords", []):
            low = kw.lower()
            if low not in seen_kw:
                seen_kw.add(low)
                all_kw.append(kw)
    merged["mood_keywords"] = all_kw[:8]

    # Visual narrative: pick longest
    merged["visual_narrative"] = max(
        (r.get("visual_narrative", "") or "" for r in results), key=len
    )

    # Inspiration notes: pick longest
    merged["inspiration_notes"] = max(
        (r.get("inspiration_notes", "") or "" for r in results), key=len
    )

    # Design direction: pick longest
    merged["design_direction"] = max(
        (r.get("design_direction", "") or "" for r in results), key=len
    ) or None

    # Color psychology: merge all entries
    psych: dict = {}
    for r in results:
        for hex_val, info in (r.get("color_psychology") or {}).items():
            if hex_val not in psych:
                psych[hex_val] = info
    merged["color_psychology"] = psych or None

    # Design system: pick the one with most keys
    best_ds = max(results, key=lambda r: len(r.get("design_system") or {}))
    merged["design_system"] = best_ds.get("design_system")

    # Anti-patterns: union, deduplicated
    all_ap: list[str] = []
    seen_ap: set[str] = set()
    for r in results:
        for ap in r.get("anti_patterns") or []:
            low = ap.lower()
            if low not in seen_ap:
                seen_ap.add(low)
                all_ap.append(ap)
    merged["anti_patterns"] = all_ap[:5] or None

    # Image search terms: union, deduplicated
    all_img: list[str] = []
    seen_img: set[str] = set()
    for r in results:
        for term in r.get("image_search_terms") or []:
            low = term.lower()
            if low not in seen_img:
                seen_img.add(low)
                all_img.append(term)
    merged["image_search_terms"] = all_img[:4] or None

    return merged


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

MOODBOARD_PROMPT = """You are an elite creative director with deep expertise in visual design, brand strategy, color theory, and typography. Build a comprehensive mood board specification.

**Project:** {name}
**Concept:** {description}
**Existing Palette:** {colors}
**Style Keywords:** {keywords}
**Brief:** {style_notes}

Be specific, bold, and opinionated. Reference real design movements and concrete visual references. No generic advice.

Provide:

1. **Typography** (3 fonts): Specific Google Fonts that work as a hierarchy. For each, explain WHY it fits (e.g. "Playfair Display — high-contrast serifs evoke editorial luxury"). Include heading, body, and accent roles. For each font, also provide a short evocative sample headline (3-6 words) that showcases the font's personality.

2. **Extended Palette** (3-4 colors): Complementary hex colors with purpose. Specify each color's role (accent, background, text, highlight) and give it a descriptive name.

3. **Mood Keywords** (5-7): Go beyond surface adjectives. Use evocative, synesthetic descriptors (e.g. "hand-thrown ceramic warmth" not "warm", "brutalist concrete grid" not "structured").

4. **Visual Narrative** (3-4 sentences): A vivid creative brief referencing specific textures, photographic styles, spatial qualities, and cultural touchstones.

5. **Design System**: Corner radius style, shadow approach, layout density, imagery style (photo vs illustration, saturated vs muted), iconography style.

6. **Color Psychology**: For each color (existing + new), describe emotional effect AND suggest a pairing partner with context.

7. **Anti-patterns** (2-3): What to explicitly avoid with this aesthetic.

8. **Image Search Terms** (3-4): Unsplash-style search keywords for finding reference imagery that matches this mood (e.g. "minimalist architecture fog", "hand-dyed indigo textile closeup").

Respond ONLY with valid JSON:
{{
  "typography": [
    {{"font": "Font Name", "role": "heading", "rationale": "why this font", "sample_text": "Short Evocative Headline"}},
    {{"font": "Font Name", "role": "body", "rationale": "why", "sample_text": "Words That Flow"}},
    {{"font": "Font Name", "role": "accent", "rationale": "why", "sample_text": "Bold Statement"}}
  ],
  "additional_colors": [
    {{"hex": "#hexcode", "role": "accent", "name": "Descriptive Name"}},
    {{"hex": "#hexcode", "role": "background", "name": "Descriptive Name"}}
  ],
  "mood_keywords": ["evocative keyword 1", "evocative keyword 2"],
  "visual_narrative": "3-4 sentence vivid creative brief",
  "design_system": {{
    "corner_radius": "description",
    "shadows": "description",
    "layout_density": "description",
    "imagery_style": "description",
    "iconography": "description"
  }},
  "color_psychology": {{
    "#hexcode": {{"meaning": "emotional effect", "pair_with": "#hexcode2", "pair_context": "why this pairing"}}
  }},
  "anti_patterns": ["avoid this", "avoid that"],
  "image_search_terms": ["evocative search term 1", "evocative search term 2", "evocative search term 3"],
  "inspiration_notes": "same as visual_narrative for backward compat",
  "design_direction": "1-2 sentence summary of design system notes"
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

    # Ensure LLM generates color psychology for ALL palette colors, not just new ones
    if request.colors:
        formatted_prompt += f"\n\nIMPORTANT: Your color_psychology object MUST include entries for ALL of these existing palette colors: {colors_str} — plus any new colors you propose in additional_colors."

    # Add document context if "Use Docs as Inspiration" is enabled
    if request.use_existing_docs:
        try:
            from backend.services.rag import get_rag_service
            rag = get_rag_service()
            search_query = f"{request.name} {' '.join(request.keywords or [])} design style theme"
            results = await rag.search(
                query=search_query,
                collection_filter=request.collection_filters[0] if request.collection_filters else None,
                limit=request.max_doc_sources,
                enhance_query=request.enhance_query,
                skip_cache=request.skip_cache,
                intelligence_level=request.intelligence_level,
                folder_id=request.folder_id,
                include_subfolders=request.include_subfolders,
            )
            if results:
                doc_snippets = "\n---\n".join([r.get("content", "")[:500] for r in results[:request.max_doc_sources]])
                formatted_prompt += f"\n\nDocument Context (use these as inspiration for design direction, themes, and content):\n{doc_snippets}"
                logger.info("Added document context to moodboard prompt", num_chunks=len(results))
        except Exception as e:
            logger.warning("Failed to fetch document context for moodboard", error=str(e))

    # Inject CoT / verification instructions (gated by model tier)
    try:
        from backend.services.session_memory import get_model_tier
        _resolved_model = model_name or llm_config.default_model
        _tier_info = get_model_tier(_resolved_model)
        _is_small = _tier_info["tier"] in ("tiny", "small")
    except Exception:
        _is_small = False

    if not _is_small:
        if request.enable_cot:
            formatted_prompt += (
                "\n\nThink step-by-step before generating your design suggestions: "
                "1) Analyze the provided colors and keywords for mood/tone. "
                "2) Consider how typography, additional colors, and design direction align with the concept. "
                "3) Generate cohesive, well-reasoned suggestions."
            )
        if request.enable_verification and request.use_existing_docs:
            formatted_prompt += (
                "\n\nIMPORTANT: Verify your suggestions against the provided Document Context. "
                "Ensure design direction, themes, and keywords are grounded in the source documents. "
                "Do NOT fabricate themes that contradict the document content."
            )

    # Get LLM(s) and execute
    try:
        models_used: list[str] = []

        if request.multi_llm and len(request.multi_llm.providers) > 1:
            # ── Multi-LLM: call providers in parallel ──
            provider_configs: list[tuple[str, Optional[str]]] = []
            for p in request.multi_llm.providers:
                p_type = llm_config.default_provider
                p_model = p.get("model")
                if p.get("provider_id"):
                    try:
                        prov = await LLMProviderService.get_provider(db, p["provider_id"])
                        if prov:
                            p_type = prov.provider_type
                            p_model = p_model or prov.default_chat_model
                    except Exception:
                        pass
                provider_configs.append((p_type, p_model))

            logger.info("Multi-LLM moodboard generation", providers=len(provider_configs),
                        strategy=request.multi_llm.merge_strategy)

            raw_results = await asyncio.gather(
                *[_call_single_llm(pt, pm, formatted_prompt, temperature_override=request.temperature_override) for pt, pm in provider_configs],
                return_exceptions=True,
            )

            valid_results = [r for r in raw_results if isinstance(r, dict)]
            models_used = [f"{pt}/{pm}" for (pt, pm), r in zip(provider_configs, raw_results)
                           if isinstance(r, dict)]

            if not valid_results:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="All LLM providers failed to generate suggestions",
                )

            if request.multi_llm.merge_strategy == "merge":
                suggestions_data = _merge_suggestions(valid_results)
            else:
                # best_of: pick the most complete response
                suggestions_data = max(valid_results, key=_score_suggestion)

        else:
            # ── Single LLM call ──
            result = await _call_single_llm(provider_type, model_name, formatted_prompt, temperature_override=request.temperature_override)
            if result is None:
                # Fallback
                result = {
                    "typography": ["Inter", "Playfair Display", "Space Grotesk"],
                    "additional_colors": ["#f97316", "#22c55e", "#0ea5e9"],
                    "mood_keywords": ["modern", "elegant", "sophisticated"],
                    "inspiration_notes": f"Based on the {request.name} concept with {keywords_str} aesthetic, this mood board suggests a contemporary direction with the provided color palette.",
                    "design_direction": "Focus on clean lines and balanced compositions.",
                    "color_psychology": {},
                }
            suggestions_data = result
            models_used = [f"{provider_type}/{model_name}"]

        # Normalize and build response
        suggestions_data = _normalize_suggestions(suggestions_data)

        # Post-process: ensure all palette colors have psychology entries
        psych = suggestions_data.get("color_psychology") or {}
        for color in (request.colors or []):
            if color and color not in psych:
                psych[color] = {"meaning": "User-selected palette color", "pair_with": None, "pair_context": None}
        suggestions_data["color_psychology"] = psych if psych else None

        suggestions = GeneratedSuggestions(
            typography=suggestions_data.get("typography", ["Inter", "Roboto", "Open Sans"]),
            additional_colors=suggestions_data.get("additional_colors", ["#f97316", "#22c55e", "#0ea5e9"]),
            mood_keywords=suggestions_data.get("mood_keywords", ["modern", "clean", "professional"]),
            inspiration_notes=suggestions_data.get("inspiration_notes", "A sophisticated visual direction."),
            design_direction=suggestions_data.get("design_direction"),
            color_psychology=suggestions_data.get("color_psychology"),
            visual_narrative=suggestions_data.get("visual_narrative"),
            design_system=suggestions_data.get("design_system"),
            anti_patterns=suggestions_data.get("anti_patterns"),
            image_search_terms=suggestions_data.get("image_search_terms"),
        )

        board_id = str(uuid4())

        logger.info(
            "Mood board generated successfully",
            board_id=board_id,
            models=models_used,
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
            model_used=" + ".join(models_used),
        )

    except Exception as e:
        logger.error(
            "Mood board generation failed",
            name=request.name,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Mood board generation failed"
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
    if suggestions:
        suggestions = _normalize_suggestions(suggestions)

    color_palette = board_data.get("colors", [])
    if suggestions.get("_colors_flat"):
        existing = {c.lower() for c in color_palette}
        new_colors = [c for c in suggestions["_colors_flat"] if c.lower() not in existing]
        color_palette = color_palette + new_colors[:max(0, 8 - len(color_palette))]
    elif suggestions.get("additional_colors"):
        color_palette = color_palette + suggestions.get("additional_colors", [])

    style_tags = board_data.get("keywords", [])
    if suggestions.get("mood_keywords"):
        existing = {k.lower() for k in style_tags}
        new_kw = [k for k in suggestions["mood_keywords"] if k.lower() not in existing]
        style_tags = style_tags + new_kw

    # Extract flat font names from rich typography
    themes = suggestions.get("_typography_flat", suggestions.get("typography", []))

    # Create the MoodBoard record
    mood_board = MoodBoard(
        user_id=user.user_id,
        name=name,
        description=description,
        prompt=prompt,
        status=MoodBoardStatus.COMPLETED.value,
        images=[],
        themes=themes,
        color_palette=color_palette,
        style_tags=style_tags,
        generated_suggestions=suggestions,
        canvas_data=board_data.get("canvas_data"),
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
                "color_palette": b.color_palette or [],
                "themes": b.themes or [],
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
        "generated_suggestions": board.generated_suggestions or {},
        "canvas_data": board.canvas_data,
        "thumbnail_url": board.thumbnail_url,
        "is_public": board.is_public,
        "created_at": board.created_at.isoformat(),
        "updated_at": board.updated_at.isoformat(),
    }


class MoodBoardUpdateRequest(BaseModel):
    """Request to update a mood board."""
    name: Optional[str] = None
    description: Optional[str] = None
    color_palette: Optional[List[str]] = None
    themes: Optional[List[str]] = None
    style_tags: Optional[List[str]] = None
    generated_suggestions: Optional[Dict[str, Any]] = None
    canvas_data: Optional[Dict[str, Any]] = None


@router.patch("/{board_id}")
async def update_mood_board(
    board_id: str,
    request: MoodBoardUpdateRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update a mood board's editable fields.
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

    # Apply updates
    if request.name is not None:
        board.name = request.name
    if request.description is not None:
        board.description = request.description
    if request.color_palette is not None:
        board.color_palette = request.color_palette
    if request.themes is not None:
        board.themes = request.themes
    if request.style_tags is not None:
        board.style_tags = request.style_tags
    if request.generated_suggestions is not None:
        board.generated_suggestions = request.generated_suggestions
    if request.canvas_data is not None:
        board.canvas_data = request.canvas_data

    await db.commit()
    await db.refresh(board)

    logger.info("Mood board updated", board_id=board_id, user_id=user.user_id)

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
        "generated_suggestions": board.generated_suggestions or {},
        "canvas_data": board.canvas_data,
        "thumbnail_url": board.thumbnail_url,
        "is_public": board.is_public,
        "created_at": board.created_at.isoformat(),
        "updated_at": board.updated_at.isoformat(),
    }


class CanvasUpdateRequest(BaseModel):
    """Lightweight request for auto-saving canvas positions."""
    canvas_data: Dict[str, Any]


@router.put("/{board_id}/canvas")
async def update_canvas(
    board_id: str,
    request: CanvasUpdateRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Quick-save canvas layout without touching other fields. Designed for auto-save."""
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

    board.canvas_data = request.canvas_data
    await db.commit()

    return {"saved": True, "board_id": board_id}


@router.post("/document-preview")
async def preview_documents(
    user: AuthenticatedUser,
    name: str = "",
    keywords: List[str] = Query(default=[]),
    collection_filters: Optional[List[str]] = Query(default=None),
    folder_id: Optional[str] = None,
    include_subfolders: bool = False,
    max_sources: int = Query(default=5, ge=1, le=20),
):
    """Preview which documents will be used as inspiration before generating."""
    try:
        from backend.services.rag import get_rag_service
        rag = get_rag_service()
        search_query = f"{name} {' '.join(keywords)} design style theme"
        results = await rag.search(
            query=search_query,
            collection_filter=collection_filters[0] if collection_filters else None,
            limit=max_sources,
            folder_id=folder_id,
            include_subfolders=include_subfolders,
        )
        docs = []
        if results:
            for r in results[:max_sources]:
                docs.append({
                    "title": r.get("source", r.get("document_name", "Unknown")),
                    "snippet": r.get("content", "")[:300],
                    "score": round(r.get("score", 0.0), 3),
                    "doc_id": r.get("document_id", ""),
                })
        return {"documents": docs, "total_found": len(docs)}
    except Exception as e:
        logger.error("Document preview failed", error=str(e))
        return {"documents": [], "total_found": 0, "error": "Document preview failed"}


@router.post("/{board_id}/enhance")
async def enhance_mood_board(
    board_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Generate/regenerate AI suggestions for an existing mood board.
    Useful for boards created before the generated_suggestions feature,
    or to refresh suggestions with a new LLM call.
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

    # Build prompt from existing board data
    colors_str = ", ".join(board.color_palette or []) or "Not specified"
    keywords_str = ", ".join(board.style_tags or []) or "Not specified"

    formatted_prompt = MOODBOARD_PROMPT.format(
        name=board.name,
        description=board.description or "Not specified",
        colors=colors_str,
        keywords=keywords_str,
        style_notes="Enhance this existing mood board with rich design insights",
    )

    # Get LLM and generate
    provider_type = llm_config.default_provider
    model_name = None

    try:
        llm = LLMFactory.get_chat_model(
            provider=provider_type,
            model=model_name,
            temperature=0.8,
            max_tokens=2048,
        )

        response = await llm.ainvoke(formatted_prompt)
        output = response.content

        # Parse JSON response
        output_clean = output.strip()
        if output_clean.startswith("```json"):
            output_clean = output_clean[7:]
        if output_clean.startswith("```"):
            output_clean = output_clean[3:]
        if output_clean.endswith("```"):
            output_clean = output_clean[:-3]

        try:
            suggestions_data = json.loads(output_clean.strip())
        except json.JSONDecodeError:
            suggestions_data = {
                "typography": board.themes or ["Inter", "Playfair Display", "Space Grotesk"],
                "additional_colors": ["#f97316", "#22c55e", "#0ea5e9"],
                "mood_keywords": ["modern", "elegant", "sophisticated"],
                "inspiration_notes": f"A visual exploration of the {board.name} concept, blending {keywords_str} aesthetics.",
                "design_direction": "Focus on clean lines and balanced compositions with the existing palette.",
                "color_psychology": {},
            }

        # Normalize the new format
        suggestions_data = _normalize_suggestions(suggestions_data)

        # Update the board
        board.generated_suggestions = suggestions_data
        # Clear canvas_data so frontend re-runs autoLayoutMoodboard() with new suggestions
        # (old canvas persists outdated titles, missing sticky notes, floating blocks)
        board.canvas_data = None

        # Also update themes if they were empty
        if not board.themes and suggestions_data.get("_typography_flat"):
            board.themes = suggestions_data["_typography_flat"]
        elif not board.themes and suggestions_data.get("typography"):
            board.themes = suggestions_data["typography"]

        # Add extra colors (cap at 8 total, case-insensitive dedup)
        extra_colors = suggestions_data.get("_colors_flat", []) or suggestions_data.get("additional_colors", [])
        if extra_colors and board.color_palette:
            existing = {c.lower() for c in board.color_palette}
            new_colors = [c for c in extra_colors if isinstance(c, str) and c.lower() not in existing]
            remaining = 8 - len(board.color_palette)
            if new_colors and remaining > 0:
                board.color_palette = board.color_palette + new_colors[:remaining]

        # Add keywords (case-insensitive dedup)
        if suggestions_data.get("mood_keywords") and board.style_tags:
            existing = {k.lower() for k in board.style_tags}
            new_kw = [k for k in suggestions_data["mood_keywords"] if k.lower() not in existing]
            if new_kw:
                board.style_tags = board.style_tags + new_kw

        await db.commit()
        await db.refresh(board)

        logger.info("Mood board enhanced", board_id=board_id, user_id=user.user_id)

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
            "generated_suggestions": board.generated_suggestions or {},
            "canvas_data": board.canvas_data,
            "thumbnail_url": board.thumbnail_url,
            "is_public": board.is_public,
            "created_at": board.created_at.isoformat(),
            "updated_at": board.updated_at.isoformat(),
        }

    except Exception as e:
        logger.error("Mood board enhancement failed", board_id=board_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Enhancement failed"
        )


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
