"""
PPTX Format Generator

Full implementation of PowerPoint presentation generation.
Migrated from generator.py for modularity.
"""

import os
import re
import zipfile
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

import structlog
from lxml import etree

from ..base import BaseFormatGenerator
from ..factory import register_generator
from ...models import OutputFormat, GenerationJob, Section
from ...config import THEMES, FONT_FAMILIES, LAYOUT_TEMPLATES

if TYPE_CHECKING:
    from ...template_analyzer import TemplateAnalysis

logger = structlog.get_logger(__name__)


# =============================================================================
# Utility Functions (module-level)
# =============================================================================

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_theme_colors(theme_key: str = "business", custom_colors: dict = None) -> dict:
    """Get theme colors, with fallback to business theme."""
    theme = THEMES.get(theme_key, THEMES["business"]).copy()
    if custom_colors:
        for key in ["primary", "secondary", "accent", "text", "background"]:
            if key in custom_colors:
                theme[key] = custom_colors[key]
    return theme


def strip_markdown(text: str) -> str:
    """Remove markdown formatting, returning clean text."""
    if not text:
        return ""
    # Remove headers (at start of line or anywhere - catches ### in bullets)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Also remove ### anywhere in content (LLM sometimes outputs "### Title" in bullets)
    text = re.sub(r'\s*#{1,6}\s+', ' ', text)
    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove code blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Clean up multiple spaces that might result
    text = re.sub(r'  +', ' ', text)
    return text


def sanitize_text(text: str) -> str:
    """Sanitize text for PPTX XML compatibility."""
    if not text:
        return ""
    # Remove ASCII control chars except tab, newline, carriage return
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = text.replace('\x0b', ' ')
    text = text.replace('\x0c', ' ')
    return text


def is_sentence_complete(text: str) -> bool:
    """
    Check if a sentence appears to be complete (not truncated mid-thought).

    Returns False if the text:
    - Ends with a conjunction (and, or, but)
    - Ends with a preposition (to, for, of, in, on, at, by, with)
    - Ends with an article (the, a, an)
    - Ends with a verb suggesting continuation (is, are, was, were, will, can, should)
    - Appears to be cut off mid-phrase
    """
    if not text or len(text) < 5:
        return False

    text = text.strip()

    # Re-check length after strip (whitespace-only strings become empty)
    if not text:
        return False

    # Check for proper sentence ending
    if text[-1] in '.!?':
        return True

    # Colon at end indicates incomplete (expects continuation)
    if text[-1] == ':':
        return False

    # Get the last word
    words = text.split()
    if not words:
        return False

    last_word = words[-1].lower().rstrip('.,;:!?-–—')

    # Words that indicate incomplete thought
    incomplete_endings = {
        # Conjunctions
        'and', 'or', 'but', 'nor', 'yet', 'so',
        # Prepositions
        'to', 'for', 'of', 'in', 'on', 'at', 'by', 'with', 'from', 'into', 'about',
        'between', 'through', 'during', 'before', 'after', 'above', 'below', 'using',
        'including', 'featuring', 'providing', 'offering', 'creating', 'developing',
        # Articles
        'the', 'a', 'an',
        # Verbs suggesting continuation (present participle / gerunds)
        'is', 'are', 'was', 'were', 'will', 'can', 'could', 'should', 'would', 'may', 'might',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'allowing', 'enabling', 'showcasing', 'demonstrating', 'driving', 'boosting',
        'leveraging', 'turns', 'showcase', 'existing', 'leading', 'growing', 'building',
        # Adjectives that typically precede nouns
        'customized', 'enhanced', 'innovative', 'strategic', 'comprehensive', 'integrated',
        'unique', 'exclusive', 'interactive', 'immersive', 'engaging', 'dynamic',
        'new', 'additional', 'various', 'specific', 'particular', 'increased', 'brand',
        'influential', 'emotional', 'favorite', 'effective', 'affordable', 'exclusive',
        # Other incomplete indicators
        'that', 'which', 'who', 'whom', 'whose', 'this', 'these', 'those',
        'such', 'make', 'making', 'highlight', 'highlighting', 'amplify', 'amplifying',
        'like', 'as', 'its', 'their', 'our', 'your',
        # Proper nouns that often need context (Real Madrid, NIVEA, etc)
        'real', 'nivea',
        # Hyphenated compound words that are often incomplete
        'in-stadium', 'in-store', 'on-site', 'off-site', 'co-branded',
        # PHASE 9 FIX: Additional incomplete endings found in generated content
        # Verbs commonly truncated
        'tap', 'create', 'develop', 'build', 'enhance', 'leverage', 'drive', 'increase',
        'boost', 'reach', 'engage', 'connect', 'promote', 'deliver', 'transform',
        'maximize', 'optimize', 'establish', 'strengthen', 'generate', 'foster',
        # Present participles commonly truncated
        'tapping', 'reaching', 'engaging', 'connecting', 'promoting', 'delivering',
        'transforming', 'maximizing', 'optimizing', 'establishing', 'strengthening',
        'generating', 'fostering', 'targeting', 'expanding', 'increasing',
        # More adjectives
        'global', 'local', 'premium', 'authentic', 'memorable', 'impactful',
        'significant', 'powerful', 'extensive', 'notable', 'key', 'major', 'primary',
    }

    if last_word in incomplete_endings:
        return False

    # Check for pattern: ends with "for X" where X is a number (incomplete phrase like "for 3")
    if len(words) >= 2:
        second_last = words[-2].lower().rstrip('.,;:!?-–—')
        if second_last in ['for', 'by', 'with', 'over', 'under', 'about', 'around']:
            # Check if last word is a number or short word
            if last_word.isdigit() or len(last_word) <= 2:
                return False

    return True


def ensure_complete_thought(text: str, max_chars: int) -> str:
    """
    Truncate text while ensuring it remains a complete thought.

    Instead of just cutting at word boundaries, this function:
    1. Tries to find a natural sentence boundary
    2. If not possible, restructures the text to be complete
    3. Removes trailing incomplete phrases
    """
    if not text or len(text) <= max_chars:
        return text

    text = text.strip()

    # Strategy 1: Find complete sentence boundary within limit
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = ""
    for sent in sentences:
        if len(result) + len(sent) + 1 <= max_chars:
            result = (result + " " + sent).strip() if result else sent
        else:
            break

    if result and is_sentence_complete(result):
        return result

    # Strategy 2: Find clause boundary (;, –, —, :)
    for sep in ['; ', ' – ', ' — ', ': ', ', ']:
        parts = text.split(sep)
        candidate = parts[0].strip()
        if len(candidate) <= max_chars and is_sentence_complete(candidate):
            return candidate

    # Strategy 3: Truncate at word boundary and remove incomplete endings
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.5:
        truncated = truncated[:last_space]

    # Aggressively remove incomplete trailing phrases
    words = truncated.split()
    incomplete_endings = {
        'and', 'or', 'but', 'nor', 'yet', 'so', 'to', 'for', 'of', 'in', 'on', 'at',
        'by', 'with', 'from', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'will',
        'can', 'could', 'should', 'would', 'that', 'which', 'who', 'this', 'these',
        'highlight', 'highlighting', 'amplify', 'amplifying', 'make', 'making',
        'using', 'including', 'featuring', 'providing', 'offering', 'creating', 'developing',
        'customized', 'enhanced', 'innovative', 'strategic', 'comprehensive', 'integrated',
        'unique', 'exclusive', 'interactive', 'immersive', 'engaging', 'dynamic',
        'new', 'additional', 'various', 'specific', 'particular', 'increased', 'brand',
        'like', 'as', 'its', 'their', 'our', 'your', 'such',
        # Additional incomplete endings
        'allowing', 'enabling', 'showcasing', 'demonstrating', 'driving', 'boosting',
        'leveraging', 'turns', 'showcase', 'existing', 'leading', 'growing', 'building',
        'influential', 'emotional', 'favorite', 'effective', 'affordable',
        'real', 'nivea',  # Brand names that need context
        'in-stadium', 'in-store', 'on-site', 'off-site', 'co-branded',
    }

    # Remove words until we have a complete thought
    while len(words) > 3:
        last_word = words[-1].lower().rstrip('.,;:!?-–— ')
        if last_word in incomplete_endings:
            words.pop()
        else:
            break

    result = ' '.join(words).rstrip('.,;:!?-–— ')

    # Final check - if still incomplete, try to fix by removing trailing incomplete words
    if not is_sentence_complete(result):
        # Try to complete common incomplete patterns
        result_words = result.split()
        if result_words and len(result_words) > 1:  # Need at least 2 words to remove one
            last_word = result_words[-1].lower()
            if last_word in ['and', 'or']:
                result = result.rsplit(' ', 1)[0]  # Remove the conjunction
            elif last_word in ['for', 'with', 'to']:
                result = result.rsplit(' ', 1)[0]  # Remove the preposition

    return result[:max_chars]


async def llm_condense_text(text: str, max_chars: int, context: dict = None) -> str:
    """Use LLM to condense text while preserving meaning.

    Args:
        text: Original text to condense
        max_chars: Maximum allowed characters
        context: Optional dict with surrounding context for coherent condensing:
            - slide_title: The title of the current slide
            - prev_bullet: The previous bullet point (for flow)
            - next_bullet: The next bullet point (for flow)
    """
    if len(text) <= max_chars:
        return text

    try:
        from backend.services.llm import EnhancedLLMFactory

        llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="content_generation",
            prefer_fast=True,
        )

        # Build context section if provided
        context_section = ""
        if context:
            slide_title = context.get('slide_title', '')
            prev_bullet = context.get('prev_bullet', '')
            next_bullet = context.get('next_bullet', '')

            context_parts = []
            if slide_title:
                context_parts.append(f"Slide topic: {slide_title}")
            if prev_bullet:
                prev_display = prev_bullet[:50] + "..." if len(prev_bullet) > 50 else prev_bullet
                context_parts.append(f"Previous point: {prev_display}")
            if next_bullet:
                next_display = next_bullet[:50] + "..." if len(next_bullet) > 50 else next_bullet
                context_parts.append(f"Next point: {next_display}")

            if context_parts:
                context_section = f"""
Context (ensure logical flow):
{chr(10).join('- ' + p for p in context_parts)}
"""

        prompt = f"""Condense this text to fit within {max_chars} characters while preserving key information:

{text}
{context_section}
Requirements:
- Keep numbers, percentages, and statistics exactly
- Preserve the main point
- Avoid repeating information from adjacent points
- Ensure logical flow with surrounding content
- Output ONLY the condensed text, nothing else
- Must be under {max_chars} characters"""

        response = await llm.ainvoke(prompt)
        # Safely handle None content
        if response and hasattr(response, 'content') and response.content:
            condensed = response.content.strip()
            if len(condensed) <= max_chars:
                return condensed
        # If no valid content, fall through to fallback

    except Exception as e:
        logger.warning(
            f"LLM condensation failed: {e}. "
            f"Text length: {len(text)}, max: {max_chars}. Using smart truncation fallback."
        )

    # Fallback: use ensure_complete_thought for smart truncation
    # This ensures we don't create incomplete sentences
    return ensure_complete_thought(text, max_chars)


async def llm_rewrite_for_slide(
    text: str,
    max_chars: int,
    text_type: str = "bullet",
    enable_llm_rewrite: bool = True,
    context: dict = None,
) -> str:
    """
    Use LLM to intelligently rewrite text to fit slide constraints.

    This is called when text exceeds max_chars and needs intelligent condensing.
    The LLM understands presentation context and produces proper slide-ready text.

    Args:
        text: Original text that's too long
        max_chars: Maximum allowed characters
        text_type: Type of text (title, bullet, sub-bullet) for context
        enable_llm_rewrite: Whether to use LLM rewriting (from settings)
        context: Optional dict with surrounding context for coherent rewriting:
            - slide_title: The title of the current slide
            - prev_bullet: The previous bullet point (for flow)
            - next_bullet: The next bullet point (for flow)
            - all_bullets: List of all bullets on this slide (for deduplication)

    Returns:
        Rewritten text that fits within max_chars
    """
    if not text or len(text) <= max_chars:
        return text

    # If LLM rewrite is disabled, use sync fallback
    if not enable_llm_rewrite:
        return enforce_text_length_sync(text, max_chars, text_type)

    try:
        from backend.services.llm import EnhancedLLMFactory

        llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="content_generation",
            prefer_fast=True,  # Use faster model for quick rewrites
        )

        # Different prompts for different text types - ENTERPRISE CONSULTING STANDARDS
        if text_type == "title":
            # ACTION TITLE: McKinsey/BCG standard - title should be the key takeaway
            prompt = f"""Rewrite this slide title as an ACTION TITLE (consulting standard).

An ACTION TITLE states the key takeaway or insight, NOT just the topic.
- BAD: "Market Analysis" (just a topic)
- GOOD: "US Market Growing 15% Annually" (the key insight)
- BAD: "Revenue Overview"
- GOOD: "Revenue Increased 25% Year-over-Year"

Original ({len(text)} chars): "{text}"

STRICT Rules:
- Must be {max_chars} characters or less
- Must state the KEY FINDING or RECOMMENDATION, not just topic
- Include specific numbers, percentages, or outcomes when available
- Start with strong noun or result (avoid starting with "The", "A", "An")
- NO generic titles: "Overview", "Summary", "Introduction", "Analysis"
- Professional consulting tone (McKinsey/BCG style)
- No trailing punctuation except ? for questions

Return ONLY the rewritten action title, nothing else."""

        else:  # bullet or sub-bullet
            # ENTERPRISE BULLET: 6x6 rule compliant, action-oriented
            # Build context section if provided
            context_section = ""
            if context:
                slide_title = context.get('slide_title', '')
                prev_bullet = context.get('prev_bullet', '')
                next_bullet = context.get('next_bullet', '')

                context_parts = []
                if slide_title:
                    context_parts.append(f"Slide title: {slide_title}")
                if prev_bullet:
                    # Truncate long previous bullets for context
                    prev_display = prev_bullet[:60] + "..." if len(prev_bullet) > 60 else prev_bullet
                    context_parts.append(f"Previous point: {prev_display}")
                if next_bullet:
                    next_display = next_bullet[:60] + "..." if len(next_bullet) > 60 else next_bullet
                    context_parts.append(f"Next point: {next_display}")

                if context_parts:
                    context_section = f"""
SLIDE CONTEXT (ensure logical flow with surrounding points):
{chr(10).join('- ' + p for p in context_parts)}

"""

            prompt = f"""Rewrite this presentation bullet point (consulting standard).

Enterprise bullet rules (McKinsey 6x6 style):
- Maximum 6-8 words (you MUST be concise)
- Start with action verb (Increase, Launch, Reduce, Improve, etc.)
- State ONE clear insight or action per bullet
- Include specific data when available
{context_section}
Original ({len(text)} chars): "{text}"

STRICT Rules:
- Must be {max_chars} characters or less AND under 8 words
- Keep the KEY INSIGHT or main point
- Preserve numbers, percentages, specific data
- Start with action verb or key noun
- ONE complete thought per bullet
- NO trailing "..." or incomplete thoughts
- NO filler words (very, really, actually, basically)
- Professional consulting tone
- Ensure this point flows logically from previous and leads to next

Return ONLY the rewritten bullet point, nothing else."""

        response = await llm.ainvoke(prompt)
        # Safely handle None content
        if response and hasattr(response, 'content') and response.content:
            rewritten = response.content.strip().strip('"\'')

            # Verify it fits
            if len(rewritten) <= max_chars:
                logger.info(
                    f"LLM rewrote {text_type}",
                    original=len(text),
                    rewritten=len(rewritten),
                    max=max_chars,
                )
                return rewritten
            else:
                # LLM didn't follow constraint, fall back to smart truncation
                logger.warning(
                    f"LLM rewrite still too long ({len(rewritten)} > {max_chars}), using fallback"
                )
        # If no valid content, fall through to fallback

    except Exception as e:
        logger.warning(
            f"LLM rewrite failed for {text_type}: {e}. "
            f"Text length: {len(text)}, max: {max_chars}. Using smart truncation fallback."
        )

    # Fallback to synchronous smart truncation (never adds "...")
    return enforce_text_length_sync(text, max_chars, text_type)


def enforce_text_length_sync(text: str, max_chars: int, text_type: str = "text") -> str:
    """
    Synchronous fallback: Ensure text fits within max_chars while preserving meaning.

    Used when LLM rewrite fails or is not available.
    Delegates to ensure_complete_thought for smart truncation that maintains sentence integrity.
    """
    if not text or len(text) <= max_chars:
        return text

    # Use the comprehensive ensure_complete_thought function
    return ensure_complete_thought(text.strip(), max_chars)


def validate_content_quality(bullets: list, slide_title: str = "") -> tuple:
    """
    Validate content quality and return issues with suggestions.

    Returns:
        Tuple of (is_valid, issues_list, fixed_bullets)
    """
    issues = []
    fixed_bullets = []

    for i, bullet in enumerate(bullets):
        bullet_text = bullet[0] if isinstance(bullet, tuple) else bullet
        bullet_level = bullet[1] if isinstance(bullet, tuple) else 0

        # Skip None or empty bullets
        if not bullet_text:
            continue

        # Check for incomplete sentences
        if not is_sentence_complete(bullet_text):
            # Safely truncate for logging (handle short strings)
            display_text = bullet_text[:40] + "..." if len(bullet_text) > 40 else bullet_text
            issues.append(f"Bullet {i+1} appears incomplete: '{display_text}'")

            # Try to fix by removing incomplete ending
            fixed = ensure_complete_thought(bullet_text, len(bullet_text))
            if is_sentence_complete(fixed):
                if isinstance(bullet, tuple):
                    fixed_bullets.append((fixed, bullet_level))
                else:
                    fixed_bullets.append(fixed)
            else:
                # Keep original if fix didn't work
                fixed_bullets.append(bullet)
        else:
            fixed_bullets.append(bullet)

    is_valid = len(issues) == 0
    return is_valid, issues, fixed_bullets


def enforce_text_length(text: str, max_chars: int, text_type: str = "text") -> str:
    """
    Synchronous wrapper - use enforce_text_length_sync or call llm_rewrite_for_slide async.

    This is for backwards compatibility. For best results, use the async version.
    """
    return enforce_text_length_sync(text, max_chars, text_type)


# =============================================================================
# Native Bullet Formatting Functions (Enterprise Quality)
# =============================================================================

def add_native_bullet_to_paragraph(p, text: str, level: int = 0, bullet_char: str = None):
    """
    Add native PowerPoint bullet formatting to a paragraph.

    This uses OOXML a:buChar elements instead of manual text prefixes,
    which is the correct way to format bullets in PowerPoint for:
    - Proper theme integration
    - Accessibility compliance
    - Consistent visual hierarchy
    - Better editing experience

    Args:
        p: python-pptx paragraph object
        text: The bullet text (without bullet character prefix)
        level: Bullet level (0=main, 1=sub, 2=sub-sub)
        bullet_char: Optional custom bullet character (default based on level)
    """
    from pptx.oxml.ns import qn
    from lxml import etree

    # Set paragraph level for indentation
    p.level = level

    # Set the text WITHOUT manual bullet character prefix
    p.text = text

    try:
        # Access the paragraph XML element
        pPr = p._p.get_or_add_pPr()

        # Remove any existing bullet formatting
        for old_bu in list(pPr):
            tag = old_bu.tag if hasattr(old_bu, 'tag') else ''
            if 'bu' in tag.lower():
                pPr.remove(old_bu)

        # Set proper margins for bullet-to-text spacing (CRITICAL for professional look)
        # marL = left margin (where text starts from left edge)
        # indent = hanging indent (negative pulls bullet left of text)
        # Values in EMU: 914400 EMU = 1 inch, 457200 EMU = 0.5 inch
        base_margin = 457200  # 0.5 inch base margin
        level_margin = base_margin + (level * 365760)  # Add 0.4" per level
        hanging_indent = -228600  # -0.25 inch hanging indent (pulls bullet left)

        pPr.set('marL', str(level_margin))
        pPr.set('indent', str(hanging_indent))

        # Determine bullet character based on level
        if bullet_char is None:
            # Default bullet hierarchy:
            # Level 0: filled circle (•)
            # Level 1: en-dash (–)
            # Level 2: small square (▪)
            level_bullets = ['•', '–', '▪']
            bullet_char = level_bullets[min(level, len(level_bullets) - 1)]

        # Add bullet font (controls bullet character rendering)
        buFont = etree.SubElement(pPr, qn('a:buFont'))
        buFont.set('typeface', 'Arial')
        buFont.set('panose', '020B0604020202020204')

        # Add bullet character
        buChar = etree.SubElement(pPr, qn('a:buChar'))
        buChar.set('char', bullet_char)

        logger.debug(f"Applied native bullet '{bullet_char}' at level {level} with marL={level_margin}, indent={hanging_indent}")

    except Exception as e:
        logger.warning(f"Native bullet formatting failed, will use text prefix: {e}")
        # Fallback: prepend bullet character to text (existing behavior)
        p.text = f"{bullet_char} {text}"


def enable_text_autofit(text_frame, min_font_scale: int = 50, max_line_reduction: int = 20):
    """
    Enable PowerPoint's native text auto-fit to prevent text overflow.

    Instead of truncating text with "...", this allows PowerPoint to
    automatically scale font size down to fit content within the text box.
    This is the professional approach used by enterprise tools like Aspose.

    Args:
        text_frame: python-pptx TextFrame object
        min_font_scale: Minimum font scale percentage (50 = 50%, can go down to 50%)
        max_line_reduction: Maximum line spacing reduction percentage (20 = 20%)

    OOXML Output:
        <a:bodyPr>
            <a:normAutofit fontScale="50000" lnSpcReduction="20000"/>
        </a:bodyPr>
    """
    from pptx.oxml.ns import qn
    from lxml import etree

    try:
        # Access the text body XML element
        txBody = text_frame._txBody
        bodyPr = txBody.bodyPr

        if bodyPr is None:
            logger.warning("No bodyPr element found in text frame")
            return False

        # Remove any existing autofit elements (noAutofit, spAutoFit, normAutofit)
        autofit_tags = [qn('a:noAutofit'), qn('a:spAutoFit'), qn('a:normAutofit')]
        for child in list(bodyPr):
            if child.tag in autofit_tags:
                bodyPr.remove(child)

        # Add normAutofit for "Shrink text on overflow" behavior
        normAutofit = etree.SubElement(bodyPr, qn('a:normAutofit'))
        # Values are in 1/1000 of a percent (50000 = 50%, 20000 = 20%)
        normAutofit.set('fontScale', str(min_font_scale * 1000))
        normAutofit.set('lnSpcReduction', str(max_line_reduction * 1000))

        logger.debug(f"Enabled text autofit: min_scale={min_font_scale}%, max_reduction={max_line_reduction}%")
        return True

    except Exception as e:
        logger.warning(f"Failed to enable text autofit: {e}")
        return False


def find_placeholder_by_idx(slide, target_idx: int):
    """
    Find a placeholder by its index (idx), not by type.

    This is the CORRECT way to match placeholders according to OOXML spec.
    Placeholder inheritance works by idx matching between layout and slide:
        Layout placeholder idx=10 → Slide placeholder idx=10

    Args:
        slide: python-pptx Slide object
        target_idx: The placeholder index to find

    Returns:
        Placeholder shape or None
    """
    try:
        for placeholder in slide.placeholders:
            if placeholder.placeholder_format.idx == target_idx:
                return placeholder
    except Exception as e:
        logger.debug(f"Error finding placeholder by idx {target_idx}: {e}")
    return None


def get_layout_placeholder_mapping(layout) -> dict:
    """
    Extract placeholder index mapping from a slide layout.

    Returns a dict mapping placeholder type to idx for proper content targeting.

    Args:
        layout: python-pptx SlideLayout object

    Returns:
        Dict like {PP_PLACEHOLDER.TITLE: 0, PP_PLACEHOLDER.BODY: 1, ...}
    """
    mapping = {}
    try:
        for ph in layout.placeholders:
            ph_type = ph.placeholder_format.type
            ph_idx = ph.placeholder_format.idx
            mapping[ph_type] = ph_idx
            logger.debug(f"Layout placeholder: type={ph_type}, idx={ph_idx}")
    except Exception as e:
        logger.debug(f"Error extracting placeholder mapping: {e}")
    return mapping


def enforce_bullet_constraints(
    content: str,
    max_bullet_chars: int = 120,  # PHASE 11: Increased from 70 to allow complete sentences
    max_bullets: int = 6,  # ENTERPRISE STANDARD: 6x6 rule
    max_words_per_bullet: int = 12,  # PHASE 11: Increased from 8 to allow complete thoughts
    max_sub_bullets_per_main: int = 2,  # ENTERPRISE STANDARD: limit sub-bullets
) -> str:
    """
    ENTERPRISE-GRADE ENFORCEMENT: Ensure bullet content meets consulting standards.

    Implements the "6x6 Rule" used by McKinsey/BCG:
    - Max 6 main bullets per slide
    - Max ~6 words per bullet (we use 8 for flexibility)
    - Max 2 sub-bullets per main bullet
    - Each bullet should be a complete, actionable insight

    Args:
        content: Raw content with bullet points
        max_bullet_chars: Maximum characters per bullet (backup limit)
        max_bullets: Maximum number of main bullets (default: 6 per 6x6 rule)
        max_words_per_bullet: Maximum words per bullet (default: 8)
        max_sub_bullets_per_main: Maximum sub-bullets per main bullet

    Returns:
        Content with enforced enterprise constraints
    """
    if not content:
        return content

    def count_words(text: str) -> int:
        """Count words in text, excluding common stop words for accuracy."""
        words = text.split()
        return len(words)

    def truncate_to_words(text: str, max_words: int) -> str:
        """Truncate text to maximum word count while keeping it coherent."""
        words = text.split()
        if len(words) <= max_words:
            return text

        # Truncate to max_words and ensure it ends cleanly
        truncated = ' '.join(words[:max_words])

        # Remove trailing incomplete words (articles, prepositions)
        incomplete = ['the', 'a', 'an', 'to', 'for', 'of', 'in', 'on', 'at', 'and', 'or', 'but', 'with', 'by']
        words_result = truncated.split()
        while words_result and words_result[-1].lower() in incomplete:
            words_result.pop()

        return ' '.join(words_result) if words_result else truncated

    lines = content.split('\n')
    result_lines = []
    main_bullet_count = 0
    current_sub_bullets = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect if this is a main bullet or sub-bullet
        is_sub_bullet = line.startswith('  ') or stripped.startswith('◦') or stripped.startswith('-')
        is_main_bullet = stripped.startswith('•') or stripped.startswith('*') or (
            not is_sub_bullet and not stripped.startswith(' ')
        )

        if is_main_bullet:
            main_bullet_count += 1
            current_sub_bullets = 0

            if main_bullet_count > max_bullets:
                # ENTERPRISE: Stop at max bullets (6x6 rule)
                logger.info(
                    f"6x6 Rule: Limited to {max_bullets} bullets (had more)",
                    rule="enterprise_6x6",
                )
                break

            # Clean the bullet text
            bullet_text = stripped.lstrip('•*- ')

            # ENTERPRISE: Enforce word count FIRST, then char limit as backup
            word_count = count_words(bullet_text)
            if word_count > max_words_per_bullet:
                original_text = bullet_text
                bullet_text = truncate_to_words(bullet_text, max_words_per_bullet)
                logger.debug(
                    f"6x6 Rule: Truncated bullet from {word_count} to {count_words(bullet_text)} words",
                    original=original_text[:50],
                )

            # Backup: character limit for very long words
            if len(bullet_text) > max_bullet_chars:
                bullet_text = enforce_text_length(bullet_text, max_bullet_chars, "bullet")

            result_lines.append(f"• {bullet_text}")

        elif is_sub_bullet and main_bullet_count > 0:
            current_sub_bullets += 1
            if current_sub_bullets <= max_sub_bullets_per_main:
                # Clean sub-bullet
                sub_text = stripped.lstrip('◦○•*- ')

                # ENTERPRISE: Word limit for sub-bullets (slightly fewer than main)
                if count_words(sub_text) > max_words_per_bullet - 2:
                    sub_text = truncate_to_words(sub_text, max_words_per_bullet - 2)

                # Backup: character limit
                if len(sub_text) > max_bullet_chars - 10:
                    sub_text = enforce_text_length(sub_text, max_bullet_chars - 10, "sub-bullet")

                result_lines.append(f"  ◦ {sub_text}")
            elif current_sub_bullets == max_sub_bullets_per_main + 1:
                logger.debug(
                    f"6x6 Rule: Limited sub-bullets to {max_sub_bullets_per_main}",
                    rule="enterprise_6x6",
                )

    return '\n'.join(result_lines)


# =============================================================================
# PPTEval-Style Quality Scoring (Enterprise Quality Assurance)
# =============================================================================

class SlideQualityChecker:
    """
    PPTEval-style quality checker for generated slides.

    Implements automated quality scoring based on best practices from:
    - PPTAgent's PPTEval scoring system
    - McKinsey/BCG presentation standards
    - WCAG accessibility guidelines

    Checks:
    1. Content quality (bullet count, word count, completeness)
    2. Design quality (overlaps, spacing, alignment)
    3. Theme compliance (colors, fonts, consistency)
    """

    def __init__(self, max_bullets: int = 6, max_words_per_bullet: int = 8):
        self.max_bullets = max_bullets
        self.max_words_per_bullet = max_words_per_bullet
        self.issues = []

    def check_slide(self, slide) -> list:
        """
        Check a single slide for quality issues.

        Args:
            slide: python-pptx Slide object

        Returns:
            List of issue descriptions (empty if no issues)
        """
        issues = []

        try:
            # Check 1: Bullet count (6x6 rule)
            bullet_count = self._count_bullets(slide)
            if bullet_count > self.max_bullets:
                issues.append(f"Too many bullets: {bullet_count} (max {self.max_bullets})")

            # Check 2: Word count per bullet
            word_issues = self._check_word_counts(slide)
            issues.extend(word_issues)

            # Check 3: Text overflow
            overflow_issues = self._check_text_overflow(slide)
            issues.extend(overflow_issues)

            # Check 4: Shape overlaps
            overlap_issues = self._check_shape_overlaps(slide)
            issues.extend(overlap_issues)

            # Check 5: Empty shapes (excluding decorative elements)
            empty_issues = self._check_empty_shapes(slide)
            issues.extend(empty_issues)

            # Check 6: WCAG contrast compliance
            contrast_issues = self._check_wcag_contrast(slide)
            issues.extend(contrast_issues)

        except Exception as e:
            logger.warning(f"Quality check failed: {e}")

        return issues

    def check_presentation(self, presentation) -> dict:
        """
        Check entire presentation for quality issues.

        Args:
            presentation: python-pptx Presentation object

        Returns:
            Dict with 'scores' and 'issues' per slide
        """
        results = {
            'total_issues': 0,
            'slides': {},
            'scores': {
                'content': 10.0,
                'design': 10.0,
                'overall': 10.0,
            }
        }

        for idx, slide in enumerate(presentation.slides):
            slide_issues = self.check_slide(slide)
            results['slides'][idx + 1] = slide_issues
            results['total_issues'] += len(slide_issues)

        # Calculate scores (start at 10, deduct for issues)
        if results['total_issues'] > 0:
            content_penalty = min(5.0, results['total_issues'] * 0.5)
            design_penalty = min(5.0, results['total_issues'] * 0.3)
            results['scores']['content'] = max(0, 10.0 - content_penalty)
            results['scores']['design'] = max(0, 10.0 - design_penalty)
            results['scores']['overall'] = (results['scores']['content'] + results['scores']['design']) / 2

        return results

    def _count_bullets(self, slide) -> int:
        """Count total bullets across all text frames."""
        count = 0
        try:
            for shape in slide.shapes:
                if hasattr(shape, 'text_frame'):
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        # Count if it starts with bullet char or has level > 0
                        if text and (text[0] in '•◦▪●○■□–-' or para.level == 0):
                            if len(text) > 3:  # Ignore very short text
                                count += 1
        except Exception:
            pass
        return count

    def _check_word_counts(self, slide) -> list:
        """Check word counts in bullets."""
        issues = []
        try:
            for shape in slide.shapes:
                if hasattr(shape, 'text_frame'):
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            # Remove bullet character for word count
                            clean_text = text.lstrip('•◦▪●○■□–- ')
                            word_count = len(clean_text.split())
                            if word_count > self.max_words_per_bullet + 4:
                                issues.append(
                                    f"Bullet too long: {word_count} words (max {self.max_words_per_bullet})"
                                )
        except Exception:
            pass
        return issues

    def _check_text_overflow(self, slide) -> list:
        """Check for potential text overflow (heuristic check)."""
        issues = []
        try:
            for shape in slide.shapes:
                if hasattr(shape, 'text_frame') and hasattr(shape, 'height'):
                    tf = shape.text_frame
                    # Estimate text height (rough calculation)
                    total_lines = 0
                    for para in tf.paragraphs:
                        if para.text:
                            # Rough estimate: 60 chars per line
                            lines_needed = max(1, len(para.text) / 60)
                            total_lines += lines_needed

                    # Check if estimated lines exceed shape capacity
                    # Assume ~20pt line height on average
                    if shape.height:
                        from pptx.util import Pt
                        max_lines = shape.height.pt / 22  # ~22pt per line with spacing
                        if total_lines > max_lines * 1.2:
                            issues.append(
                                f"Potential overflow in '{shape.name}': ~{int(total_lines)} lines, fits ~{int(max_lines)}"
                            )
        except Exception:
            pass
        return issues

    def _check_shape_overlaps(self, slide) -> list:
        """Check for overlapping shapes (text/image overlaps)."""
        issues = []
        try:
            shapes_with_bounds = []
            for shape in slide.shapes:
                if hasattr(shape, 'left') and hasattr(shape, 'width'):
                    bounds = {
                        'name': getattr(shape, 'name', 'Shape'),
                        'left': shape.left,
                        'top': shape.top,
                        'right': shape.left + shape.width,
                        'bottom': shape.top + shape.height,
                        'is_text': hasattr(shape, 'text_frame'),
                        'is_picture': shape.shape_type.name == 'PICTURE' if hasattr(shape, 'shape_type') else False,
                    }
                    shapes_with_bounds.append(bounds)

            # Check for text/picture overlaps
            for i, shape1 in enumerate(shapes_with_bounds):
                for shape2 in shapes_with_bounds[i+1:]:
                    if shape1['is_text'] and shape2['is_picture'] or shape1['is_picture'] and shape2['is_text']:
                        # Check if bounding boxes overlap
                        if (shape1['left'] < shape2['right'] and shape1['right'] > shape2['left'] and
                            shape1['top'] < shape2['bottom'] and shape1['bottom'] > shape2['top']):
                            # Calculate overlap area
                            overlap_width = min(shape1['right'], shape2['right']) - max(shape1['left'], shape2['left'])
                            overlap_height = min(shape1['bottom'], shape2['bottom']) - max(shape1['top'], shape2['top'])
                            if overlap_width > 0 and overlap_height > 0:
                                from pptx.util import Inches
                                overlap_in = overlap_width / Inches(1)
                                if overlap_in > 0.1:  # Only report significant overlaps
                                    issues.append(
                                        f"Overlap: '{shape1['name']}' and '{shape2['name']}' ({overlap_in:.2f}in)"
                                    )
        except Exception:
            pass
        return issues

    def _check_empty_shapes(self, slide) -> list:
        """Check for empty text shapes (excluding decorative elements)."""
        issues = []
        try:
            for shape in slide.shapes:
                if hasattr(shape, 'text_frame'):
                    name = getattr(shape, 'name', '').lower()
                    # Skip decorative/branding elements
                    if any(x in name for x in ['footer', 'header', 'logo', 'line', 'rect', 'shape']):
                        continue
                    # Check if text frame is empty
                    text = ''.join(p.text for p in shape.text_frame.paragraphs).strip()
                    if not text and 'title' in name.lower():
                        issues.append(f"Empty title placeholder: '{shape.name}'")
        except Exception:
            pass
        return issues

    def _check_wcag_contrast(self, slide) -> list:
        """Check text/background contrast meets WCAG 2.1 AA requirements."""
        issues = []
        try:
            # Default background (assume white if not specified)
            default_bg = (255, 255, 255)

            for shape in slide.shapes:
                if hasattr(shape, 'text_frame'):
                    for para in shape.text_frame.paragraphs:
                        if not para.text.strip():
                            continue

                        # Get font size (default to 16pt)
                        font_size = 16
                        is_bold = False
                        text_color = None

                        try:
                            if para.font.size:
                                font_size = para.font.size.pt
                            if para.font.bold:
                                is_bold = True
                            if para.font.color and para.font.color.rgb:
                                rgb = para.font.color.rgb
                                text_color = (rgb.red, rgb.green, rgb.blue)
                        except Exception:
                            pass

                        # Skip if no explicit color (uses theme)
                        if text_color is None:
                            continue

                        # Check contrast against assumed background
                        result = validate_wcag_contrast(
                            text_color,
                            default_bg,
                            font_size_pt=int(font_size),
                            is_bold=is_bold
                        )

                        if not result['passes_aa']:
                            shape_name = getattr(shape, 'name', 'Shape')
                            issues.append(
                                f"WCAG contrast fail in '{shape_name}': {result['ratio']}:1 "
                                f"(need {result['required_ratio']}:1)"
                            )
        except Exception:
            pass
        return issues


# =============================================================================
# WCAG 2.1 Accessibility Compliance (Enterprise Requirement)
# =============================================================================

def validate_wcag_contrast(
    text_color: tuple,
    bg_color: tuple,
    font_size_pt: int = 16,
    is_bold: bool = False
) -> dict:
    """
    Validate that text/background colors meet WCAG 2.1 AA requirements.

    WCAG 2.1 Level AA requires:
    - Normal text (<18pt or <14pt bold): 4.5:1 contrast ratio minimum
    - Large text (≥18pt or ≥14pt bold): 3:1 contrast ratio minimum

    Args:
        text_color: RGB tuple (r, g, b) where each value is 0-255
        bg_color: RGB tuple (r, g, b) where each value is 0-255
        font_size_pt: Font size in points
        is_bold: Whether text is bold

    Returns:
        Dict with 'ratio', 'passes_aa', 'required_ratio', 'recommendation'
    """
    def relative_luminance(rgb: tuple) -> float:
        """Calculate relative luminance per WCAG 2.1 formula."""
        r, g, b = rgb[0] / 255, rgb[1] / 255, rgb[2] / 255

        # Apply gamma correction
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4

        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Calculate luminance for both colors
    l1 = relative_luminance(text_color)
    l2 = relative_luminance(bg_color)

    # Calculate contrast ratio (lighter over darker)
    ratio = (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

    # Determine required ratio based on text size
    # Large text: 18pt+ OR 14pt+ bold
    is_large_text = font_size_pt >= 18 or (font_size_pt >= 14 and is_bold)
    min_ratio = 3.0 if is_large_text else 4.5

    passes = ratio >= min_ratio

    return {
        'ratio': round(ratio, 2),
        'passes_aa': passes,
        'required_ratio': min_ratio,
        'is_large_text': is_large_text,
        'recommendation': (
            f"Contrast {ratio:.1f}:1 {'PASSES' if passes else 'FAILS'} WCAG AA "
            f"({'large' if is_large_text else 'normal'} text requires {min_ratio}:1)"
        )
    }


def suggest_accessible_color(
    text_color: tuple,
    bg_color: tuple,
    required_ratio: float = 4.5
) -> tuple:
    """
    Suggest an accessible color adjustment when contrast fails WCAG.

    Attempts to darken/lighten text color to meet contrast requirements
    while staying close to original appearance.

    Args:
        text_color: Original RGB tuple
        bg_color: Background RGB tuple
        required_ratio: Target contrast ratio

    Returns:
        Adjusted RGB tuple that meets requirements
    """
    def get_luminance(rgb: tuple) -> float:
        r, g, b = rgb[0] / 255, rgb[1] / 255, rgb[2] / 255
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    bg_lum = get_luminance(bg_color)
    text_lum = get_luminance(text_color)

    # Determine if we need to lighten or darken
    if bg_lum > 0.5:
        # Light background - darken text
        factor = 0.9  # Start darkening
        direction = -1
    else:
        # Dark background - lighten text
        factor = 1.1  # Start lightening
        direction = 1

    adjusted = list(text_color)

    for _ in range(50):  # Max 50 iterations
        # Adjust color
        for i in range(3):
            adjusted[i] = min(255, max(0, int(adjusted[i] * factor)))

        # Check contrast
        result = validate_wcag_contrast(tuple(adjusted), bg_color)
        if result['passes_aa']:
            return tuple(adjusted)

        # Increase adjustment
        if direction > 0:
            factor += 0.05
        else:
            factor -= 0.05

        # Prevent infinite loop
        if factor <= 0.1 or factor >= 3.0:
            break

    # Fallback: return black or white based on background
    return (0, 0, 0) if bg_lum > 0.5 else (255, 255, 255)


# =============================================================================
# Vertical Text Alignment (Professional Polish)
# =============================================================================

def set_vertical_alignment(text_frame, alignment: str = "middle"):
    """
    Set vertical text alignment within a shape.

    Args:
        text_frame: python-pptx TextFrame object
        alignment: 'top', 'middle', or 'bottom'

    PowerPoint vertical anchors:
    - MSO_ANCHOR.TOP: Text anchored to top
    - MSO_ANCHOR.MIDDLE: Text centered vertically
    - MSO_ANCHOR.BOTTOM: Text anchored to bottom
    """
    try:
        from pptx.enum.text import MSO_ANCHOR

        alignment_map = {
            'top': MSO_ANCHOR.TOP,
            'middle': MSO_ANCHOR.MIDDLE,
            'bottom': MSO_ANCHOR.BOTTOM,
        }

        anchor = alignment_map.get(alignment.lower(), MSO_ANCHOR.MIDDLE)
        text_frame.vertical_anchor = anchor
        logger.debug(f"Set vertical alignment to {alignment}")
        return True

    except Exception as e:
        logger.warning(f"Failed to set vertical alignment: {e}")
        return False


# =============================================================================
# Dynamic Font Sizing (Prevents Overflow Without Truncation)
# =============================================================================

def calculate_optimal_font_size(
    text: str,
    box_width_inches: float,
    box_height_inches: float,
    max_font: int = 24,
    min_font: int = 12,
    chars_per_inch_at_12pt: float = 8.0,
    line_height_factor: float = 1.4
) -> int:
    """
    Calculate optimal font size that fits text in container without overflow.

    Uses geometric calculation to estimate the best font size before
    rendering, avoiding the need for truncation.

    Args:
        text: The text content to fit
        box_width_inches: Container width in inches
        box_height_inches: Container height in inches
        max_font: Maximum allowed font size
        min_font: Minimum allowed font size
        chars_per_inch_at_12pt: Characters per inch at 12pt (tunable)
        line_height_factor: Line height multiplier (1.4 = 140% of font size)

    Returns:
        Optimal font size in points
    """
    import math

    if not text or box_width_inches <= 0 or box_height_inches <= 0:
        return max_font

    text_len = len(text)

    # Scale chars per inch based on font size
    # At 12pt, we use the base value. Larger fonts = fewer chars per inch
    def chars_per_inch(font_pt: int) -> float:
        return chars_per_inch_at_12pt * (12 / font_pt)

    # Binary search for optimal font size
    low, high = min_font, max_font
    optimal = min_font

    for _ in range(10):  # Max 10 iterations for binary search
        mid = (low + high) // 2

        # Calculate how many chars fit per line at this font size
        cpi = chars_per_inch(mid)
        chars_per_line = int(box_width_inches * cpi)

        if chars_per_line <= 0:
            high = mid - 1
            continue

        # Calculate lines needed
        lines_needed = math.ceil(text_len / chars_per_line)

        # Calculate height needed (font size * line height factor)
        line_height_pt = mid * line_height_factor
        height_needed_pt = lines_needed * line_height_pt
        height_available_pt = box_height_inches * 72  # Convert inches to points

        if height_needed_pt <= height_available_pt:
            # This font size fits, try larger
            optimal = mid
            low = mid + 1
        else:
            # Too big, try smaller
            high = mid - 1

        if low > high:
            break

    return max(min_font, min(max_font, optimal))


# =============================================================================
# Enhanced Speaker Notes with Accessibility Descriptions
# =============================================================================

def add_accessible_speaker_notes(
    slide,
    notes: str,
    describe_images: bool = True,
    include_reading_order: bool = False
):
    """
    Add speaker notes with WCAG accessibility enhancements.

    Enterprise presentations should include:
    - Presenter notes for talking points
    - Image descriptions for accessibility
    - Optionally, reading order information

    Args:
        slide: python-pptx Slide object
        notes: Speaker notes text
        describe_images: Include descriptions for images in notes
        include_reading_order: Include shape reading order info
    """
    try:
        full_notes = notes.strip() if notes else ""

        if describe_images:
            image_descriptions = []

            for shape in slide.shapes:
                # Check if shape is an image
                is_picture = (
                    hasattr(shape, 'shape_type') and
                    str(shape.shape_type).endswith('PICTURE')
                )

                if is_picture:
                    name = getattr(shape, 'name', 'Image')

                    # Try to get alt text from shape's XML element
                    alt_text = ''
                    try:
                        elem = shape._element
                        if hasattr(elem, 'nvPicPr'):
                            cNvPr = elem.nvPicPr.cNvPr
                            if cNvPr is not None:
                                alt_text = cNvPr.get('descr', '')
                    except Exception:
                        pass

                    desc = alt_text if alt_text else 'No description available'
                    image_descriptions.append(f"[{name}]: {desc}")

            if image_descriptions:
                full_notes += "\n\n--- Image Descriptions for Accessibility ---\n"
                full_notes += "\n".join(image_descriptions)

        if include_reading_order:
            full_notes += "\n\n--- Reading Order ---\n"
            for idx, shape in enumerate(slide.shapes, 1):
                name = getattr(shape, 'name', f'Shape {idx}')
                full_notes += f"{idx}. {name}\n"

        # Set notes
        notes_slide = slide.notes_slide
        notes_tf = notes_slide.notes_text_frame
        notes_tf.text = full_notes

        logger.debug(f"Added speaker notes with {len(full_notes)} chars")
        return True

    except Exception as e:
        logger.warning(f"Failed to add speaker notes: {e}")
        return False


# =============================================================================
# McKinsey-Style Action Titles
# =============================================================================

def is_action_title(title: str) -> bool:
    """
    Check if a title follows McKinsey's "action title" convention.

    Action titles state the KEY INSIGHT as a complete sentence,
    not just the topic. They should:
    - Be a complete sentence
    - State the conclusion/insight
    - Start with strong verb or clear subject

    Examples:
        BAD:  "Q3 Sales Results" (descriptive)
        GOOD: "Q3 sales exceeded targets by 15%" (action title)

    Returns:
        True if title appears to be an action title
    """
    if not title or len(title) < 10:
        return False

    title = title.strip()

    # Action titles typically have these characteristics:
    # 1. Longer than typical topic titles (usually 8+ words)
    words = title.split()
    if len(words) < 5:
        return False

    # 2. Often end with a period or contain verbs
    strong_verbs = [
        'increased', 'decreased', 'grew', 'declined', 'exceeded',
        'achieved', 'demonstrated', 'revealed', 'indicates', 'shows',
        'improved', 'reduced', 'led', 'drove', 'generated',
        'should', 'must', 'need', 'requires', 'enables'
    ]

    title_lower = title.lower()
    has_verb = any(verb in title_lower for verb in strong_verbs)

    # 3. Contains numbers or specific data points
    has_data = any(char.isdigit() for char in title) or '%' in title

    # 4. Ends with proper punctuation or has complete structure
    ends_properly = title[-1] in '.!?' or title[-1].isalnum()

    # Score: action titles typically have 2+ of these traits
    score = sum([
        len(words) >= 6,
        has_verb,
        has_data,
        ends_properly
    ])

    return score >= 2


def convert_to_action_title(title: str, content_summary: str = "") -> str:
    """
    Convert a descriptive title to an action title style.

    This is a synchronous heuristic conversion. For better results,
    use LLM-based conversion with full slide context.

    Args:
        title: Original descriptive title
        content_summary: Optional summary of slide content

    Returns:
        Suggested action title (or original if already action style)
    """
    if is_action_title(title):
        return title

    # Basic heuristic conversions for common patterns
    title_lower = title.lower().strip()

    # Pattern: "X Overview" -> "Key insights from X"
    if title_lower.endswith(' overview'):
        base = title[:-9].strip()
        return f"Key insights from {base} analysis"

    # Pattern: "X Analysis" -> "X reveals important trends"
    if title_lower.endswith(' analysis'):
        base = title[:-9].strip()
        return f"{base} reveals important patterns"

    # Pattern: "X Results" -> "X shows positive outcomes"
    if title_lower.endswith(' results'):
        base = title[:-8].strip()
        return f"{base} demonstrates measurable progress"

    # Pattern: "Summary" -> "Key takeaways for action"
    if title_lower in ['summary', 'executive summary', 'key points']:
        return "Key takeaways for immediate action"

    # Pattern: "Next Steps" -> "Recommended actions to move forward"
    if title_lower in ['next steps', 'recommendations', 'action items']:
        return "Recommended next steps for implementation"

    # Default: append action indicator
    return f"{title}: Key findings and implications"


def safe_fit_text(
    text_frame,
    font_family: str = "Calibri",
    max_size: int = 18,
    min_size: int = 10,
    bold: bool = False,
) -> bool:
    """
    Safely attempt to fit text within text frame using python-pptx fit_text().

    This is a SAFETY NET for cases where text might still overflow despite
    LLM rewriting and enforcement. It uses PowerPoint's text fitting algorithm
    to reduce font size until text fits.

    Args:
        text_frame: The TextFrame object to fit
        font_family: Font family name
        max_size: Maximum font size (will shrink from this)
        min_size: Minimum font size (won't go below this)
        bold: Whether text should be bold

    Returns:
        True if fit_text succeeded, False if it failed or wasn't applicable
    """
    try:
        # Only apply fit_text if there's substantial text that might overflow
        total_text_length = sum(len(p.text) for p in text_frame.paragraphs)
        if total_text_length < 50:
            return False  # Short text doesn't need fitting

        # Enable word wrap first (required for fit_text to work properly)
        text_frame.word_wrap = True

        # Attempt fit_text with specified constraints
        text_frame.fit_text(
            font_family=font_family,
            max_size=max_size,
            bold=bold,
        )

        # Verify font size didn't go below minimum
        for para in text_frame.paragraphs:
            if para.font.size and para.font.size.pt < min_size:
                # Font got too small, reset and log warning
                logger.warning(
                    f"fit_text reduced font below {min_size}pt, text may need condensing",
                    final_size=para.font.size.pt,
                    text_length=total_text_length,
                )
                return False

        return True

    except Exception as e:
        # fit_text can fail in various edge cases (missing fonts, etc.)
        logger.debug(f"fit_text failed (non-critical): {e}")
        return False


async def _run_vision_review(
    output_path: str,
    job: "GenerationJob",
    sections: list,
    vision_model: str = "auto",
    review_all_slides: bool = False,
) -> None:
    """
    Run optional vision-based slide review after PPTX generation.

    This renders each slide to an image and uses a vision LLM to detect
    visual issues like text overflow, poor contrast, or layout problems.

    Args:
        output_path: Path to the generated PPTX file
        job: The generation job with metadata
        sections: List of sections for content context
        vision_model: Vision model to use (auto, claude-3-sonnet, gpt-4-vision, ollama-llava)
        review_all_slides: If True, reviews all slides including title/TOC/sources
    """
    try:
        from .reviewer import VisionSlideReviewer

        # Create vision LLM function with specified model
        vision_llm_func = await _create_vision_llm_func(vision_model)
        if not vision_llm_func:
            logger.info("Vision review skipped: no vision LLM available")
            return

        reviewer = VisionSlideReviewer(
            vision_llm_func=vision_llm_func,
            render_backend="libreoffice",
        )

        logger.info(
            "Starting vision-based slide review",
            pptx_path=output_path,
            vision_model=vision_model,
            review_all_slides=review_all_slides,
        )

        # Review each slide
        all_issues = []

        # Determine which slides to review
        if review_all_slides:
            # Review all slides: title (0), TOC (1), content slides, sources
            from pptx import Presentation
            prs = Presentation(output_path)
            total_slides = len(prs.slides)
            slides_to_review = list(range(total_slides))
        else:
            # Only review content slides (skip title, TOC, sources)
            slides_to_review = [idx + 2 for idx in range(len(sections))]

        for slide_idx in slides_to_review:
            # Render slide to image
            image_path = reviewer.render_slide_to_image(output_path, slide_idx)
            if not image_path:
                logger.warning(f"Could not render slide {slide_idx} for vision review")
                continue

            # Build slide content context
            if slide_idx >= 2 and slide_idx - 2 < len(sections):
                section = sections[slide_idx - 2]
                slide_content = {
                    "title": getattr(section, "title", "") or "",
                    "bullets": section.content.split("\n") if section.content else [],
                    "has_image": bool(getattr(section, "image_path", None)),
                    "layout": getattr(section, "layout", "standard"),
                }
            else:
                # Title/TOC/Sources slides
                slide_content = {
                    "title": job.title if slide_idx == 0 else "Table of Contents" if slide_idx == 1 else "Sources",
                    "bullets": [],
                    "has_image": False,
                    "layout": "special",
                }

            # Review the slide image
            result = await reviewer.review_slide_image(
                image_path=image_path,
                slide_content=slide_content,
                slide_index=slide_idx,
            )

            if result.has_issues:
                all_issues.extend(result.issues)
                for issue in result.issues:
                    logger.warning(
                        "Vision review found issue",
                        slide=slide_idx,
                        type=issue.issue_type,
                        severity=issue.severity,
                        description=issue.description,
                        suggestion=issue.suggestion,
                    )

        if all_issues:
            logger.warning(
                "Vision review completed with issues",
                total_issues=len(all_issues),
                high_severity=sum(1 for i in all_issues if i.severity == "high"),
            )

            # PHASE 9 FIX: Apply auto-fixes for detected visual issues
            fixes_applied = await _apply_vision_review_fixes(output_path, all_issues)
            if fixes_applied > 0:
                logger.info(f"Vision review auto-fixes applied: {fixes_applied} fixes")
        else:
            logger.info("Vision review completed: no visual issues detected")

    except ImportError as e:
        logger.warning(f"Vision review not available: {e}")
    except Exception as e:
        logger.error(f"Vision review failed: {e}")


async def _apply_vision_review_fixes(pptx_path: str, issues: list) -> int:
    """
    PHASE 9 FIX: Apply auto-fixes for visual issues detected by vision review.

    This function opens the PPTX, applies fixes for issues that can be auto-corrected,
    and saves the modified file.

    Args:
        pptx_path: Path to the PPTX file to fix
        issues: List of ReviewIssue objects from vision review

    Returns:
        Number of fixes applied
    """
    from pptx import Presentation
    from pptx.util import Inches, Pt

    try:
        prs = Presentation(pptx_path)
        fixes_applied = 0

        # Group issues by slide (issue should have slide_index if available)
        # The issues come from ReviewIssue dataclass which has issue_type, description, severity, suggestion
        for issue in issues:
            try:
                issue_type = issue.issue_type
                description = issue.description.lower() if hasattr(issue, 'description') else ""

                # Fix 1: Empty placeholders / empty text shapes
                if 'empty' in description and 'placeholder' in description:
                    # Remove empty shapes from all slides
                    for slide in prs.slides:
                        shapes_to_remove = []
                        for shape in slide.shapes:
                            if hasattr(shape, 'text_frame') and hasattr(shape.text_frame, 'text'):
                                if not shape.text_frame.text.strip():
                                    shapes_to_remove.append(shape)

                        for shape in shapes_to_remove[:3]:  # Limit removals per slide
                            try:
                                sp = shape._element
                                sp.getparent().remove(sp)
                                fixes_applied += 1
                                logger.debug(f"Removed empty shape from slide")
                            except Exception:
                                pass

                # Fix 2: Font size too small - increase minimum font
                elif 'font' in description and 'small' in description:
                    for slide in prs.slides:
                        for shape in slide.shapes:
                            if hasattr(shape, 'text_frame'):
                                for para in shape.text_frame.paragraphs:
                                    if para.font.size and para.font.size < Pt(14):
                                        para.font.size = Pt(14)
                                        fixes_applied += 1
                                        logger.debug("Increased small font size to 14pt")

                # Fix 3: Overlap issues - adjust spacing (limited auto-fix capability)
                elif 'overlap' in description:
                    # Log for manual review - overlap is complex to auto-fix
                    logger.info(f"Overlap issue detected - consider manual review: {description}")
                    # Could add position adjustments here if we had specific slide/shape info

                # Fix 4: Truncated text - already handled by pre-render review
                elif 'truncat' in description:
                    logger.info(f"Truncation issue detected - content was likely too long: {description}")

            except Exception as fix_err:
                logger.debug(f"Could not apply fix for issue '{issue.issue_type}': {fix_err}")

        if fixes_applied > 0:
            prs.save(pptx_path)
            logger.info(f"Saved PPTX with {fixes_applied} vision review fixes")

        return fixes_applied

    except Exception as e:
        logger.error(f"Failed to apply vision review fixes: {e}")
        return 0


async def _create_vision_llm_func(vision_model: str = "auto"):
    """
    Create a vision LLM function for slide review.

    Args:
        vision_model: Vision model to use (auto, claude-3-sonnet, gpt-4-vision, ollama-llava)

    Returns:
        Async function that takes (prompt, image_path) and returns response string,
        or None if no vision LLM is available.
    """
    try:
        from backend.services.llm import EnhancedLLMFactory
        import base64

        async def vision_func(prompt: str, image_path: str) -> str:
            """Call vision LLM with image."""
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

            # Determine media type
            if image_path.lower().endswith(".png"):
                media_type = "image/png"
            elif image_path.lower().endswith((".jpg", ".jpeg")):
                media_type = "image/jpeg"
            else:
                media_type = "image/png"

            # Get vision-capable model based on setting
            if vision_model == "auto":
                llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                    operation="vision_review",
                    prefer_fast=True,
                )
            elif vision_model.startswith("ollama"):
                # Use Ollama vision model (e.g., llava)
                from backend.services.llm import create_ollama_vision_func
                ollama_model = vision_model.replace("ollama-", "") or "llava"
                ollama_func = create_ollama_vision_func(model=ollama_model)
                if ollama_func:
                    return await ollama_func(prompt, image_path)
                else:
                    raise ValueError(f"Ollama vision model not available: {ollama_model}")
            else:
                # Use specified model directly
                llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                    operation="vision_review",
                    model_override=vision_model,
                )

            # Create message with image
            from langchain_core.messages import HumanMessage

            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            )

            response = await llm.ainvoke([message])
            # Safely extract content from response
            if response and hasattr(response, 'content') and response.content:
                return response.content
            return ""

        return vision_func

    except Exception as e:
        logger.warning(f"Could not create vision LLM function: {e}")
        return None


# =============================================================================
# PPTX Generator
# =============================================================================

@register_generator(OutputFormat.PPTX)
class PPTXGenerator(BaseFormatGenerator):
    """Full implementation of PowerPoint presentation generator.

    Features:
    - Template support with theme extraction
    - Professional styling with multiple themes
    - Slide animations and transitions
    - Chart and table integration
    - Speaker notes
    - Image insertion
    - TOC with hyperlinks to slides
    """

    @property
    def format_name(self) -> str:
        return "pptx"

    @property
    def file_extension(self) -> str:
        return ".pptx"

    def _extract_theme_from_template(self, template_path: str, prs) -> dict:
        """Extract actual colors and fonts from template using XML parsing."""
        extracted = {
            "primary": None,
            "secondary": None,
            "accent1": None,
            "accent2": None,
            "accent3": None,
            "accent4": None,
            "accent5": None,
            "accent6": None,
            "text": "333333",
            "background": "FFFFFF",
            "font_heading": None,
            "font_body": None,
            "bullet_color": None,
            "default_shape_fill": None,
            "font_alt": None,
        }

        try:
            with zipfile.ZipFile(template_path, 'r') as zf:
                theme_xml = zf.read('ppt/theme/theme1.xml')
                root = etree.fromstring(theme_xml)
                ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}

                # Extract color scheme
                color_scheme = root.find('.//a:clrScheme', ns)
                if color_scheme is not None:
                    color_map = {
                        'dk1': 'text',
                        'lt1': 'background',
                        'dk2': 'secondary',
                        'accent1': 'primary',
                        'accent2': 'accent1',
                        'accent3': 'accent2',
                        'accent4': 'accent3',
                        'accent5': 'accent4',
                        'accent6': 'accent5',
                    }
                    for color_elem in color_scheme:
                        tag = color_elem.tag.split('}')[-1]
                        if tag in color_map:
                            srgb = color_elem.find('.//a:srgbClr', ns)
                            if srgb is not None:
                                extracted[color_map[tag]] = srgb.get('val')
                            else:
                                sys_clr = color_elem.find('.//a:sysClr', ns)
                                if sys_clr is not None:
                                    last_clr = sys_clr.get('lastClr')
                                    if last_clr:
                                        extracted[color_map[tag]] = last_clr
                        if tag.startswith('accent'):
                            srgb = color_elem.find('.//a:srgbClr', ns)
                            if srgb is not None:
                                extracted[tag] = srgb.get('val')

                # Extract objectDefaults
                obj_defaults = root.find('.//a:objectDefaults', ns)
                if obj_defaults is not None:
                    sp_def = obj_defaults.find('.//a:spDef', ns)
                    if sp_def is not None:
                        bu_clr = sp_def.find('.//a:buClr/a:srgbClr', ns)
                        if bu_clr is not None:
                            extracted["bullet_color"] = bu_clr.get('val')
                        solid_fill = sp_def.find('.//a:spPr/a:solidFill/a:srgbClr', ns)
                        if solid_fill is not None:
                            extracted["default_shape_fill"] = solid_fill.get('val')
                        def_rpr = sp_def.find('.//a:defRPr', ns)
                        if def_rpr is not None:
                            latin = def_rpr.find('.//a:latin', ns)
                            if latin is not None and latin.get('typeface'):
                                extracted["font_alt"] = latin.get('typeface')

                # Extract font scheme
                font_scheme = root.find('.//a:fontScheme', ns)
                if font_scheme is not None:
                    major_font = font_scheme.find('.//a:majorFont/a:latin', ns)
                    minor_font = font_scheme.find('.//a:minorFont/a:latin', ns)
                    if major_font is not None:
                        extracted["font_heading"] = major_font.get('typeface')
                    if minor_font is not None:
                        extracted["font_body"] = minor_font.get('typeface')

        except Exception as e:
            logger.warning(f"Could not parse theme XML: {e}")

        # Fallback to shape analysis
        if not extracted["primary"]:
            try:
                if prs.slide_masters:
                    master = prs.slide_masters[0]
                    for shape in master.shapes:
                        if hasattr(shape, 'fill') and shape.fill.type:
                            try:
                                if shape.fill.fore_color and shape.fill.fore_color.rgb:
                                    color = str(shape.fill.fore_color.rgb)
                                    if not extracted["primary"]:
                                        extracted["primary"] = color
                                    elif not extracted["secondary"]:
                                        extracted["secondary"] = color
                            except Exception:
                                pass
            except Exception as e:
                logger.debug(f"Shape analysis fallback failed: {e}")

        return extracted

    async def generate(
        self,
        job: GenerationJob,
        filename: str,
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> str:
        """Generate a PowerPoint presentation.

        Args:
            job: The generation job containing metadata and sections
            filename: The output filename
            template_analysis: Optional template analysis for styling

        Returns:
            Path to the generated PPTX file
        """
        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt
            from pptx.dml.color import RGBColor
            from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
            from pptx.enum.shapes import MSO_SHAPE
            from pptx.opc.constants import RELATIONSHIP_TYPE as RT

            # Get output directory from config
            from ...models import GenerationConfig
            config = GenerationConfig()

            # Check if using a template PPTX
            template_path = job.metadata.get("template_pptx_path")
            use_template_styling = False

            # Standard dimensions for 16:9
            STANDARD_WIDTH = 13.333
            STANDARD_HEIGHT = 7.5

            if template_path and os.path.exists(template_path):
                prs = Presentation(template_path)
                # Remove all existing slides
                while len(prs.slides) > 0:
                    slide_id = prs.slides._sldIdLst[0].rId
                    prs.part.drop_rel(slide_id)
                    del prs.slides._sldIdLst[0]
                use_template_styling = True

                template_width = prs.slide_width.inches
                template_height = prs.slide_height.inches
                width_scale = template_width / STANDARD_WIDTH
                height_scale = template_height / STANDARD_HEIGHT

                logger.info(
                    "Using PPTX template",
                    template_path=template_path,
                    width_scale=f"{width_scale:.2f}",
                    height_scale=f"{height_scale:.2f}",
                )
            else:
                prs = Presentation()
                prs.slide_width = Inches(STANDARD_WIDTH)
                prs.slide_height = Inches(STANDARD_HEIGHT)
                width_scale = 1.0
                height_scale = 1.0

            # Scaling helper functions
            def scale_inches(value: float, axis: str = "w") -> float:
                return value * (width_scale if axis == "w" else height_scale)

            def scaled_inches(value: float, axis: str = "w"):
                return Inches(scale_inches(value, axis))

            # Get theme and font configuration
            theme_key = job.metadata.get("theme", "business")
            custom_colors = job.metadata.get("custom_colors")
            font_family_key = job.metadata.get("font_family", "modern")

            # Extract template theme if using template
            template_theme = None
            learned_layouts = None
            slide_content_planner = None

            if use_template_styling and template_path:
                template_theme = self._extract_theme_from_template(template_path, prs)
                logger.info(
                    "Extracted template theme",
                    primary=template_theme.get('primary'),
                    accent1=template_theme.get('accent1'),
                    accent2=template_theme.get('accent2'),
                    fonts=f"{template_theme.get('font_heading')}/{template_theme.get('font_body')}",
                )

                # Learn layouts from template for per-slide constraints
                # Option 3: Check per-document override first, then fall back to system config
                system_per_slide_constraints = config.enable_per_slide_constraints if hasattr(config, 'enable_per_slide_constraints') else True
                enable_per_slide_constraints = job.metadata.get('enable_per_slide_constraints', system_per_slide_constraints)
                if enable_per_slide_constraints:
                    try:
                        # Re-load presentation to analyze layouts (before slides were removed)
                        from pptx import Presentation as TemplatePrs
                        template_for_analysis = TemplatePrs(template_path)

                        from .template_learner import TemplateLayoutLearner, SlideContentPlanner

                        # Get vision analysis config - check per-document override first, then fall back to system config
                        # Option 3: Both global default in settings AND per-document override
                        system_vision_enabled = config.enable_template_vision_analysis if hasattr(config, 'enable_template_vision_analysis') else False
                        system_vision_model = config.template_vision_model if hasattr(config, 'template_vision_model') else 'auto'

                        vision_config = {
                            'enable_template_vision_analysis': job.metadata.get(
                                'enable_template_vision_analysis',
                                system_vision_enabled  # Fall back to system default
                            ),
                            'template_vision_model': job.metadata.get(
                                'template_vision_model',
                                system_vision_model  # Fall back to system default
                            ),
                        }

                        layout_learner = TemplateLayoutLearner(template_for_analysis, vision_config)

                        # Log effective vision settings (helpful for debugging)
                        if vision_config['enable_template_vision_analysis']:
                            logger.info(
                                "Template vision analysis enabled",
                                source="per-document" if 'enable_template_vision_analysis' in job.metadata else "system-config",
                                model=vision_config['template_vision_model'],
                            )

                        # Run async learning - we're already in an async context (generate is async)
                        # so we can directly await the coroutine
                        learned_layouts = await layout_learner.learn_from_template(template_path)

                        if learned_layouts:
                            slide_content_planner = SlideContentPlanner(layout_learner)
                            logger.info(
                                "Learned template layouts for per-slide constraints",
                                num_layouts=len(learned_layouts),
                                layout_names=list(learned_layouts.keys())[:5],
                            )
                    except Exception as e:
                        logger.warning(f"Template layout learning failed, using defaults: {e}")
                        learned_layouts = None
                        slide_content_planner = None

            # Configure theme based on template or custom settings
            if use_template_styling:
                if template_theme and template_theme.get("primary"):
                    theme = {
                        "primary": template_theme.get("primary", "333333"),
                        "secondary": template_theme.get("secondary", "666666"),
                        "accent": template_theme.get("accent1", "0066CC"),
                        "text": template_theme.get("text", "333333"),
                        "light_gray": "999999",
                        "slide_background": "none",
                        "header_style": "none",
                        "bullet_style": "circle",
                        "accent_position": "none",
                        "accent1": template_theme.get("accent1"),
                        "accent2": template_theme.get("accent2"),
                        "accent3": template_theme.get("accent3"),
                        "accent4": template_theme.get("accent4"),
                        "accent5": template_theme.get("accent5"),
                        "accent6": template_theme.get("accent6"),
                        "bullet_color": template_theme.get("bullet_color"),
                        "default_shape_fill": template_theme.get("default_shape_fill"),
                    }
                else:
                    theme = {
                        "primary": "333333",
                        "secondary": "666666",
                        "accent": "0066CC",
                        "text": "333333",
                        "light_gray": "999999",
                        "slide_background": "none",
                        "header_style": "none",
                        "bullet_style": "circle",
                        "accent_position": "none",
                    }
                heading_font = template_theme.get("font_heading") if template_theme else None
                body_font = template_theme.get("font_body") if template_theme else None
                if not heading_font and template_theme:
                    heading_font = template_theme.get("font_alt")
                if not body_font and template_theme:
                    body_font = template_theme.get("font_alt")
                # Fallback to safe defaults if no fonts found in template
                if not heading_font:
                    heading_font = "Calibri Light"
                if not body_font:
                    body_font = "Calibri"
            else:
                theme = get_theme_colors(theme_key, custom_colors)
                font_config = FONT_FAMILIES.get(font_family_key, FONT_FAMILIES["modern"])
                heading_font = font_config["heading"]
                body_font = font_config["body"]

            layout_key = job.metadata.get("layout", "standard")
            layout_config = LAYOUT_TEMPLATES.get(layout_key, LAYOUT_TEMPLATES["standard"])

            # Animation settings
            enable_animations = job.metadata.get("animations", False)
            animation_speed = job.metadata.get("animation_speed", "med")
            animation_duration_ms = job.metadata.get("animation_duration_ms")

            # Slide review settings (LLM-based quality review)
            # Default: ENABLED for templates (use_template_styling=True), disabled otherwise
            # Can be explicitly set via enable_slide_review or enable_quality_review in metadata
            default_review_enabled = use_template_styling  # Enable by default when using a template
            enable_slide_review = job.metadata.get(
                "enable_slide_review",
                job.metadata.get("enable_quality_review", default_review_enabled)
            )
            slide_reviewer = None
            if enable_slide_review:
                from .reviewer import SlideReviewer
                # Get LLM generate function if available
                llm_generate_func = job.metadata.get("llm_generate_func")
                slide_reviewer = SlideReviewer(llm_generate_func=llm_generate_func)
                logger.info(
                    "Slide review enabled (will review and auto-fix slides before rendering)",
                    reason="template mode" if use_template_styling else "explicit setting"
                )

            TRANSITION_DURATIONS = {
                "very_slow": 2000,
                "slow": 1500,
                "med": 750,
                "fast": 400,
                "very_fast": 200,
            }

            def get_transition_duration():
                if animation_speed == "custom" and animation_duration_ms:
                    return animation_duration_ms
                return TRANSITION_DURATIONS.get(animation_speed, 750)

            transition_duration = get_transition_duration()

            # Layout selection function with smart layout matching
            def get_slide_layout(layout_type: str = "content", has_image: bool = False):
                """Select the most appropriate layout from the template.

                Args:
                    layout_type: Type of slide - 'title', 'toc', 'content', 'conclusion', 'sources', 'contact'
                    has_image: Whether the slide will have an image

                Returns:
                    The best matching slide layout
                """
                if use_template_styling:
                    try:
                        # Build a mapping of layout names for smart selection
                        layout_names = {i: layout.name.lower() for i, layout in enumerate(prs.slide_layouts)}

                        # Define keywords for each layout type
                        layout_keywords = {
                            "title": ["title", "titel", "cover", "opening"],
                            "toc": ["agenda", "contents", "toc", "inhalt", "overview"],
                            "content_with_image": ["bild", "image", "picture", "foto", "chart"],
                            "content_text_only": ["text", "textfeld", "body", "content"],
                            "conclusion": ["fazit", "summary", "conclusion", "closing", "end"],
                            "sources": ["source", "reference", "quelle", "appendix"],
                            "contact": ["kontakt", "contact", "info"],
                        }

                        def find_layout_by_keywords(keywords: list, exclude_title: bool = True) -> int:
                            """Find a layout matching any of the keywords."""
                            for idx, name in layout_names.items():
                                # Skip title layout if we're looking for content
                                if exclude_title and idx == 0:
                                    continue
                                for keyword in keywords:
                                    if keyword in name:
                                        return idx
                            return -1

                        # Select layout based on type
                        if layout_type == "title":
                            # Title layout is always at index 0 in standard presentations
                            if len(prs.slide_layouts) > 0:
                                return prs.slide_layouts[0]
                            return None  # Will be handled by caller

                        elif layout_type == "toc":
                            idx = find_layout_by_keywords(layout_keywords["toc"])
                            if idx >= 0:
                                return prs.slide_layouts[idx]
                            # Fallback to text layout
                            idx = find_layout_by_keywords(layout_keywords["content_text_only"])
                            if idx >= 0:
                                return prs.slide_layouts[idx]

                        elif layout_type == "content":
                            if has_image:
                                idx = find_layout_by_keywords(layout_keywords["content_with_image"])
                                if idx >= 0:
                                    return prs.slide_layouts[idx]
                            # Try text-only layout
                            idx = find_layout_by_keywords(layout_keywords["content_text_only"])
                            if idx >= 0:
                                return prs.slide_layouts[idx]

                        elif layout_type == "conclusion":
                            idx = find_layout_by_keywords(layout_keywords["conclusion"])
                            if idx >= 0:
                                return prs.slide_layouts[idx]

                        elif layout_type == "sources":
                            idx = find_layout_by_keywords(layout_keywords["sources"])
                            if idx >= 0:
                                return prs.slide_layouts[idx]
                            # Fallback to text layout
                            idx = find_layout_by_keywords(layout_keywords["content_text_only"])
                            if idx >= 0:
                                return prs.slide_layouts[idx]

                        elif layout_type == "contact":
                            idx = find_layout_by_keywords(layout_keywords["contact"])
                            if idx >= 0:
                                return prs.slide_layouts[idx]

                        # Default fallback - use layout 1 for content, 0 for title
                        if layout_type == "title" and len(prs.slide_layouts) > 0:
                            return prs.slide_layouts[0]
                        elif len(prs.slide_layouts) > 1:
                            return prs.slide_layouts[1]
                        else:
                            return prs.slide_layouts[-1]

                    except Exception as e:
                        logger.warning(f"Failed to get template layout: {e}")
                        if len(prs.slide_layouts) > 0:
                            return prs.slide_layouts[-1]
                        return None  # Will be handled by caller
                else:
                    # Use blank layout (last layout) or fallback to index 6 if available
                    blank_idx = min(6, len(prs.slide_layouts) - 1)
                    return prs.slide_layouts[blank_idx]

            # Placeholder removal function
            def remove_empty_placeholders(slide, preserve_branding: bool = True, keep_picture_placeholder: bool = False):
                from pptx.enum.shapes import MSO_SHAPE_TYPE
                from pptx.enum.shapes import PP_PLACEHOLDER

                if not use_template_styling:
                    return

                REMOVABLE_PLACEHOLDER_TYPES = {
                    PP_PLACEHOLDER.BODY,
                    PP_PLACEHOLDER.SUBTITLE,
                    PP_PLACEHOLDER.FOOTER,
                    PP_PLACEHOLDER.SLIDE_NUMBER,
                    PP_PLACEHOLDER.DATE,
                    PP_PLACEHOLDER.OBJECT,
                }

                try:
                    shapes_to_remove = []
                    for shape in slide.shapes:
                        try:
                            if not hasattr(shape, 'is_placeholder') or not shape.is_placeholder:
                                continue

                            ph_type = None
                            if hasattr(shape, 'placeholder_format') and shape.placeholder_format:
                                ph_type = shape.placeholder_format.type

                            # For title placeholders, only keep if they have content
                            # Empty title placeholders should be removed (we create TextBoxes instead)
                            if ph_type == PP_PLACEHOLDER.TITLE or ph_type == PP_PLACEHOLDER.CENTER_TITLE:
                                # Check if title placeholder is empty
                                has_title_content = False
                                if hasattr(shape, 'text') and shape.text and shape.text.strip():
                                    has_title_content = True
                                if not has_title_content:
                                    shapes_to_remove.append(shape)
                                continue

                            # Keep picture placeholder if we're going to use it
                            if keep_picture_placeholder and ph_type == PP_PLACEHOLDER.PICTURE:
                                continue

                            # REMOVE picture placeholders when not being used
                            # This prevents "Insert Picture" placeholder showing on title slides
                            if ph_type == PP_PLACEHOLDER.PICTURE and not keep_picture_placeholder:
                                shapes_to_remove.append(shape)
                                continue

                            if preserve_branding and ph_type not in REMOVABLE_PLACEHOLDER_TYPES:
                                continue

                            has_content = False
                            if hasattr(shape, 'text') and shape.text and shape.text.strip():
                                has_content = True
                            if hasattr(shape, 'shape_type') and shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                                has_content = True

                            if not has_content:
                                shapes_to_remove.append(shape)

                        except (ValueError, AttributeError):
                            pass

                    for shape in shapes_to_remove:
                        try:
                            sp = shape._element
                            sp.getparent().remove(sp)
                        except Exception:
                            pass

                except Exception as e:
                    logger.warning(f"Failed to remove empty placeholders: {e}")

            # Find and use picture placeholder
            def find_picture_placeholder(slide):
                """Find picture placeholder in slide if available."""
                from pptx.enum.shapes import PP_PLACEHOLDER
                try:
                    for shape in slide.shapes:
                        if hasattr(shape, 'is_placeholder') and shape.is_placeholder:
                            if hasattr(shape, 'placeholder_format') and shape.placeholder_format:
                                if shape.placeholder_format.type == PP_PLACEHOLDER.PICTURE:
                                    return shape
                except Exception:
                    pass
                return None

            def copy_layout_branding_to_slide(slide, layout):
                """Copy embedded pictures and branding shapes from layout to slide.

                python-pptx doesn't automatically copy non-placeholder shapes from layouts.
                This function manually copies:
                - Embedded pictures (logos, backgrounds)
                - Decorative shapes (rectangles, lines) in header/footer/branding zones

                Args:
                    slide: The slide to copy branding elements to
                    layout: The slide layout to copy from
                """
                from copy import deepcopy
                from lxml import etree

                try:
                    # Access the layout's XML tree
                    layout_spTree = layout._element.cSld.spTree

                    nsmap = {
                        'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
                        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
                    }

                    # Get the slide's shape tree
                    slide_spTree = slide._element.cSld.spTree

                    # Get slide dimensions for determining branding zones
                    slide_width_emu = prs.slide_width
                    slide_height_emu = prs.slide_height

                    def is_branding_shape(shape_elem) -> bool:
                        """Check if a shape is likely a branding/decorative element."""
                        try:
                            # Get shape position from xfrm element
                            xfrm = shape_elem.find('.//a:xfrm', nsmap)
                            if xfrm is None:
                                return False

                            off = xfrm.find('a:off', nsmap)
                            ext = xfrm.find('a:ext', nsmap)
                            if off is None or ext is None:
                                return False

                            x = int(off.get('x', 0))
                            y = int(off.get('y', 0))
                            w = int(ext.get('cx', 0))
                            h = int(ext.get('cy', 0))

                            # Skip if no meaningful position/size
                            if w == 0 or h == 0:
                                return False

                            # Header zone: top 15% of slide
                            in_header = y < slide_height_emu * 0.15

                            # Footer zone: bottom 10% of slide
                            in_footer = (y + h) > slide_height_emu * 0.90

                            # Right side branding zone (logos): right 30% and top 25%
                            in_logo_zone = x > slide_width_emu * 0.70 and y < slide_height_emu * 0.25

                            # Thin horizontal lines (decorative accents)
                            is_thin_line = h < slide_height_emu * 0.02 and w > slide_width_emu * 0.15

                            # Check if it's a rectangle (common for header/footer bars)
                            prstGeom = shape_elem.find('.//a:prstGeom', nsmap)
                            is_rect = prstGeom is not None and prstGeom.get('prst') == 'rect'

                            # Branding elements are typically in header/footer/logo zones
                            # and are simple shapes (rectangles, lines)
                            if (in_header or in_footer or in_logo_zone) and (is_rect or is_thin_line):
                                return True

                            return False
                        except Exception:
                            return False

                    def get_shape_name(shape_elem, tag_type: str) -> str:
                        """Get the name of a shape from its nvSpPr or nvPicPr."""
                        try:
                            if tag_type == 'sp':
                                nvPr = shape_elem.find('.//p:nvSpPr/p:cNvPr', nsmap)
                            else:  # pic
                                nvPr = shape_elem.find('.//p:nvPicPr/p:cNvPr', nsmap)
                            return nvPr.get('name', '') if nvPr is not None else ''
                        except Exception:
                            return ''

                    def shape_exists(shape_name: str, tag_type: str) -> bool:
                        """Check if a shape with the given name already exists on the slide."""
                        if not shape_name:
                            return False
                        tag = f'p:{tag_type}'
                        existing = slide_spTree.findall(f'.//{tag}', nsmap)
                        for existing_shape in existing:
                            existing_name = get_shape_name(existing_shape, tag_type)
                            if existing_name == shape_name:
                                return True
                        return False

                    # 1. Copy pictures (logos, background images)
                    pics = layout_spTree.findall('.//p:pic', nsmap)
                    for pic in pics:
                        try:
                            shape_name = get_shape_name(pic, 'pic')

                            if shape_exists(shape_name, 'pic'):
                                continue

                            pic_copy = deepcopy(pic)

                            # Handle image relationship
                            blipFill = pic_copy.find('.//a:blipFill/a:blip', nsmap)
                            if blipFill is not None:
                                embed_attr = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'
                                rId = blipFill.get(embed_attr)
                                if rId:
                                    try:
                                        layout_part = layout.part
                                        image_part = layout_part.related_parts.get(rId)
                                        if image_part:
                                            new_rId = slide.part.relate_to(image_part, image_part.reltype)
                                            blipFill.set(embed_attr, new_rId)
                                            slide_spTree.append(pic_copy)
                                            logger.debug(f"Copied layout picture '{shape_name}' to slide")
                                    except Exception as rel_err:
                                        logger.debug(f"Could not copy image relationship for '{shape_name}': {rel_err}")

                        except Exception as pic_err:
                            logger.debug(f"Could not copy layout picture: {pic_err}")

                    # 2. Copy decorative shapes (rectangles, lines) in branding zones
                    shapes = layout_spTree.findall('.//p:sp', nsmap)
                    for shape in shapes:
                        try:
                            # Skip placeholders (they have nvPr/ph element)
                            nvPr = shape.find('.//p:nvSpPr/p:nvPr', nsmap)
                            if nvPr is not None:
                                ph = nvPr.find('p:ph', nsmap)
                                if ph is not None:
                                    continue  # Skip placeholders

                            # Check if this is a branding shape
                            if not is_branding_shape(shape):
                                continue

                            shape_name = get_shape_name(shape, 'sp')
                            if shape_exists(shape_name, 'sp'):
                                continue

                            # Deep copy the shape
                            shape_copy = deepcopy(shape)
                            slide_spTree.append(shape_copy)
                            logger.debug(f"Copied layout branding shape '{shape_name}' to slide")

                        except Exception as shape_err:
                            logger.debug(f"Could not copy layout shape: {shape_err}")

                except Exception as e:
                    logger.debug(f"Layout branding copy failed (non-fatal): {e}")

            def add_picture_with_alt_text(slide, image_path, left, top, width, height, alt_text: str = ""):
                """
                ENTERPRISE/ACCESSIBILITY: Add picture with alt text for WCAG compliance.

                Alt text is required for:
                - Screen reader accessibility
                - SEO when presentations are converted to web formats
                - Professional enterprise compliance standards

                Args:
                    slide: The slide to add the picture to
                    image_path: Path to the image file
                    left, top: Position (Inches objects)
                    width, height: Dimensions (Inches objects)
                    alt_text: Descriptive text for accessibility

                Returns:
                    The picture shape object
                """
                pic = slide.shapes.add_picture(
                    image_path,
                    left, top,
                    width=width,
                    height=height,
                )

                # Set alt text for accessibility
                if alt_text:
                    try:
                        pic.name = alt_text[:255]  # PowerPoint limits name length
                        # Set description in non-visual properties for screen readers
                        pic._element.nvPicPr.cNvPr.set('descr', alt_text[:1000])
                    except Exception as e:
                        logger.debug(f"Could not set image alt text: {e}")

                return pic

            def get_centered_image_position(target_width, target_height, content_area_top=None):
                """Calculate centered position for image within content area."""
                slide_width = prs.slide_width.inches
                slide_height = prs.slide_height.inches

                # Horizontal centering
                x = (slide_width - target_width) / 2

                # Vertical positioning: below content area or centered in remaining space
                if content_area_top is not None:
                    # Position below content area (e.g., after bullets)
                    y = content_area_top
                else:
                    # Center vertically with offset for title area
                    title_area_height = 1.5  # Approximate title height
                    available_height = slide_height - title_area_height
                    y = title_area_height + (available_height - target_height) / 2

                return Inches(x), Inches(y)

            def insert_image_into_placeholder(slide, image_path: str, placeholder_shape=None, center_image: bool = True):
                """Insert image into placeholder or add as new shape with proper sizing.

                This function ensures images:
                1. Preserve aspect ratio
                2. Fit within maximum dimensions
                3. Are positioned within slide bounds (no negative coordinates)
                """
                try:
                    # Get actual image dimensions to preserve aspect ratio
                    from PIL import Image
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size
                        # Prevent division by zero for corrupt/invalid images
                        if img_height == 0 or img_width == 0:
                            logger.warning(f"Invalid image dimensions: {img_width}x{img_height}")
                            img_aspect = 1.0  # Default to square aspect ratio
                        else:
                            img_aspect = img_width / img_height

                    # Define maximum allowed dimensions for content images
                    max_width = 5.0  # inches
                    max_height = 4.0  # inches

                    slide_width = prs.slide_width.inches
                    slide_height = prs.slide_height.inches

                    # Calculate target dimensions preserving aspect ratio
                    if img_aspect > (max_width / max_height):
                        # Image is wider - constrain by width
                        target_width = min(max_width, slide_width * 0.5)
                        target_height = target_width / img_aspect
                    else:
                        # Image is taller - constrain by height
                        target_height = min(max_height, slide_height * 0.6)
                        target_width = target_height * img_aspect

                    # Ensure dimensions don't exceed slide bounds
                    if target_width > slide_width * 0.6:
                        target_width = slide_width * 0.6
                        target_height = target_width / img_aspect
                    if target_height > slide_height * 0.6:
                        target_height = slide_height * 0.6
                        target_width = target_height * img_aspect

                    if placeholder_shape and not center_image:
                        # Use placeholder position if available and reasonable
                        ph_left = placeholder_shape.left.inches
                        ph_top = placeholder_shape.top.inches

                        # Ensure position is within bounds
                        x = max(0.3, min(ph_left, slide_width - target_width - 0.3))
                        y = max(1.0, min(ph_top, slide_height - target_height - 0.3))
                    else:
                        # Position on right side with margin, ensuring no negative coords
                        x = max(0.5, slide_width - target_width - 0.5)
                        # Scale the y position based on template height
                        # Standard slides are 7.5" tall, use proportional positioning
                        y = 2.0 * height_scale  # Below title area, scaled

                    # Ensure image doesn't overlap with footer zone
                    # Footer starts at slide_height - 0.35, leave 0.1" margin
                    footer_zone_top = slide_height - 0.45
                    max_image_bottom = footer_zone_top
                    if y + target_height > max_image_bottom:
                        # Reduce height to fit above footer
                        target_height = max_image_bottom - y
                        # Recalculate width to maintain aspect ratio
                        target_width = target_height * img_aspect
                        logger.debug(f"Image height reduced to {target_height:.2f}in to avoid footer overlap")

                    logger.debug(
                        f"Image positioning: {target_width:.1f}x{target_height:.1f} at ({x:.1f}, {y:.1f})"
                    )

                    pic = slide.shapes.add_picture(
                        image_path,
                        Inches(x),
                        Inches(y),
                        width=Inches(target_width),
                        height=Inches(target_height),
                    )

                    # ACCESSIBILITY: Add alt text for screen readers (WCAG compliance)
                    # Use section title as descriptive alt text
                    try:
                        alt_text = f"Image for: {section.title[:100]}" if hasattr(section, 'title') and section.title else "Presentation image"
                        pic.name = alt_text
                        # Also set the description in the picture's non-visual properties
                        pic._element.nvPicPr.cNvPr.set('descr', alt_text)
                    except Exception as alt_err:
                        logger.debug(f"Could not set alt text: {alt_err}")

                    # Remove the empty placeholder after adding image
                    if placeholder_shape:
                        try:
                            sp = placeholder_shape._element
                            sp.getparent().remove(sp)
                        except Exception:
                            pass

                    return pic
                except Exception as e:
                    logger.debug(f"Could not insert image: {e}")
                return None

            # Slide notes function - enhanced with accessibility support
            def add_slide_notes(slide, notes_text: str, include_image_descriptions: bool = True):
                """Add speaker notes with optional image accessibility descriptions."""
                try:
                    # Use the enhanced accessible speaker notes function
                    add_accessible_speaker_notes(
                        slide,
                        notes_text,
                        describe_images=include_image_descriptions,
                        include_reading_order=False
                    )
                except Exception as e:
                    # Fallback to basic notes if enhanced version fails
                    try:
                        notes_slide = slide.notes_slide
                        notes_frame = notes_slide.notes_text_frame
                        notes_frame.text = notes_text
                    except Exception as e2:
                        logger.warning(f"Failed to add slide notes: {e2}")

            # Slide transition function
            def add_slide_transition(slide, transition_type="fade", duration=500, speed="med"):
                try:
                    slide_elem = slide._element

                    for trans in slide_elem.findall('.//{http://schemas.openxmlformats.org/presentationml/2006/main}transition'):
                        slide_elem.remove(trans)

                    trans_elem = etree.Element(
                        '{http://schemas.openxmlformats.org/presentationml/2006/main}transition',
                        nsmap={'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}
                    )

                    ppt_speed = "slow" if speed in ("very_slow", "slow") else ("fast" if speed in ("fast", "very_fast") else "med")
                    trans_elem.set('spd', ppt_speed)
                    trans_elem.set('{http://schemas.microsoft.com/office/powerpoint/2010/main}dur', str(duration))
                    trans_elem.set('advTm', str(duration * 3))

                    if transition_type == "fade":
                        effect = etree.SubElement(trans_elem, '{http://schemas.openxmlformats.org/presentationml/2006/main}fade')
                        effect.set('thruBlk', 'true')
                    elif transition_type == "push":
                        effect = etree.SubElement(trans_elem, '{http://schemas.openxmlformats.org/presentationml/2006/main}push')
                        effect.set('dir', 'r')
                    elif transition_type == "wipe":
                        effect = etree.SubElement(trans_elem, '{http://schemas.openxmlformats.org/presentationml/2006/main}wipe')
                        effect.set('dir', 'r')
                    elif transition_type == "dissolve":
                        etree.SubElement(trans_elem, '{http://schemas.openxmlformats.org/presentationml/2006/main}dissolve')
                    elif transition_type == "blinds":
                        effect = etree.SubElement(trans_elem, '{http://schemas.openxmlformats.org/presentationml/2006/main}blinds')
                        effect.set('dir', 'vert')
                    else:
                        effect = etree.SubElement(trans_elem, '{http://schemas.openxmlformats.org/presentationml/2006/main}fade')

                    cSld = slide_elem.find('.//{http://schemas.openxmlformats.org/presentationml/2006/main}cSld')
                    if cSld is not None:
                        cSld_index = list(slide_elem).index(cSld)
                        slide_elem.insert(cSld_index + 1, trans_elem)
                    else:
                        slide_elem.append(trans_elem)

                except Exception as e:
                    logger.warning(f"Failed to add slide transition: {e}")

            # Bullet animations function
            def add_bullet_animations(slide, content_shape, animation_type="appear", delay_between=300):
                try:
                    shape_id = content_shape.shape_id
                    slide_elem = slide._element
                    tf = content_shape.text_frame
                    num_paragraphs = len([p for p in tf.paragraphs if p.text and p.text.strip()])

                    if num_paragraphs < 2:
                        return

                    timing_elem = etree.Element(
                        '{http://schemas.openxmlformats.org/presentationml/2006/main}timing',
                        nsmap={'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}
                    )

                    tnLst = etree.SubElement(timing_elem, '{http://schemas.openxmlformats.org/presentationml/2006/main}tnLst')
                    par = etree.SubElement(tnLst, '{http://schemas.openxmlformats.org/presentationml/2006/main}par')
                    cTn = etree.SubElement(par, '{http://schemas.openxmlformats.org/presentationml/2006/main}cTn')
                    cTn.set('id', '1')
                    cTn.set('dur', 'indefinite')
                    cTn.set('restart', 'never')
                    cTn.set('nodeType', 'tmRoot')

                    childTnLst = etree.SubElement(cTn, '{http://schemas.openxmlformats.org/presentationml/2006/main}childTnLst')
                    seq = etree.SubElement(childTnLst, '{http://schemas.openxmlformats.org/presentationml/2006/main}seq')
                    seq.set('concurrent', '1')
                    seq.set('nextAc', 'seek')

                    seq_cTn = etree.SubElement(seq, '{http://schemas.openxmlformats.org/presentationml/2006/main}cTn')
                    seq_cTn.set('id', '2')
                    seq_cTn.set('dur', 'indefinite')
                    seq_cTn.set('nodeType', 'mainSeq')

                    seq_childTnLst = etree.SubElement(seq_cTn, '{http://schemas.openxmlformats.org/presentationml/2006/main}childTnLst')

                    node_id = 3
                    for para_idx in range(num_paragraphs):
                        bullet_par = etree.SubElement(seq_childTnLst, '{http://schemas.openxmlformats.org/presentationml/2006/main}par')
                        bullet_cTn = etree.SubElement(bullet_par, '{http://schemas.openxmlformats.org/presentationml/2006/main}cTn')
                        bullet_cTn.set('id', str(node_id))
                        bullet_cTn.set('fill', 'hold')
                        node_id += 1

                        stCondLst = etree.SubElement(bullet_cTn, '{http://schemas.openxmlformats.org/presentationml/2006/main}stCondLst')
                        cond = etree.SubElement(stCondLst, '{http://schemas.openxmlformats.org/presentationml/2006/main}cond')

                        if para_idx == 0:
                            cond.set('evt', 'onClick')
                            cond.set('delay', '0')
                        else:
                            cond.set('evt', 'onEnd')
                            cond.set('delay', str(delay_between))

                        anim_childTnLst = etree.SubElement(bullet_cTn, '{http://schemas.openxmlformats.org/presentationml/2006/main}childTnLst')
                        effect_par = etree.SubElement(anim_childTnLst, '{http://schemas.openxmlformats.org/presentationml/2006/main}par')
                        effect_cTn = etree.SubElement(effect_par, '{http://schemas.openxmlformats.org/presentationml/2006/main}cTn')
                        effect_cTn.set('id', str(node_id))
                        effect_cTn.set('presetID', '1')
                        effect_cTn.set('presetClass', 'entr')
                        effect_cTn.set('presetSubtype', '0')
                        effect_cTn.set('fill', 'hold')
                        effect_cTn.set('nodeType', 'clickEffect')
                        node_id += 1

                        effect_stCondLst = etree.SubElement(effect_cTn, '{http://schemas.openxmlformats.org/presentationml/2006/main}stCondLst')
                        effect_cond = etree.SubElement(effect_stCondLst, '{http://schemas.openxmlformats.org/presentationml/2006/main}cond')
                        effect_cond.set('delay', '0')

                        behavior_childTnLst = etree.SubElement(effect_cTn, '{http://schemas.openxmlformats.org/presentationml/2006/main}childTnLst')
                        set_elem = etree.SubElement(behavior_childTnLst, '{http://schemas.openxmlformats.org/presentationml/2006/main}set')
                        set_cBhvr = etree.SubElement(set_elem, '{http://schemas.openxmlformats.org/presentationml/2006/main}cBhvr')
                        set_cTn = etree.SubElement(set_cBhvr, '{http://schemas.openxmlformats.org/presentationml/2006/main}cTn')
                        set_cTn.set('id', str(node_id))
                        set_cTn.set('dur', '1')
                        set_cTn.set('fill', 'hold')
                        node_id += 1

                        tgtEl = etree.SubElement(set_cBhvr, '{http://schemas.openxmlformats.org/presentationml/2006/main}tgtEl')
                        spTgt = etree.SubElement(tgtEl, '{http://schemas.openxmlformats.org/presentationml/2006/main}spTgt')
                        spTgt.set('spid', str(shape_id))
                        txEl = etree.SubElement(spTgt, '{http://schemas.openxmlformats.org/presentationml/2006/main}txEl')
                        pRg = etree.SubElement(txEl, '{http://schemas.openxmlformats.org/presentationml/2006/main}pRg')
                        pRg.set('st', str(para_idx))
                        pRg.set('end', str(para_idx))

                        attrNameLst = etree.SubElement(set_cBhvr, '{http://schemas.openxmlformats.org/presentationml/2006/main}attrNameLst')
                        attrName = etree.SubElement(attrNameLst, '{http://schemas.openxmlformats.org/presentationml/2006/main}attrName')
                        attrName.text = 'style.visibility'

                        to_elem = etree.SubElement(set_elem, '{http://schemas.openxmlformats.org/presentationml/2006/main}to')
                        strVal = etree.SubElement(to_elem, '{http://schemas.openxmlformats.org/presentationml/2006/main}strVal')
                        strVal.set('val', 'visible')

                    # Navigation conditions
                    prevCondLst = etree.SubElement(seq, '{http://schemas.openxmlformats.org/presentationml/2006/main}prevCondLst')
                    prev_cond = etree.SubElement(prevCondLst, '{http://schemas.openxmlformats.org/presentationml/2006/main}cond')
                    prev_cond.set('evt', 'onPrev')
                    prev_cond.set('delay', '0')
                    prev_tgtEl = etree.SubElement(prev_cond, '{http://schemas.openxmlformats.org/presentationml/2006/main}tgtEl')
                    etree.SubElement(prev_tgtEl, '{http://schemas.openxmlformats.org/presentationml/2006/main}sldTgt')

                    nextCondLst = etree.SubElement(seq, '{http://schemas.openxmlformats.org/presentationml/2006/main}nextCondLst')
                    next_cond = etree.SubElement(nextCondLst, '{http://schemas.openxmlformats.org/presentationml/2006/main}cond')
                    next_cond.set('evt', 'onNext')
                    next_cond.set('delay', '0')
                    next_tgtEl = etree.SubElement(next_cond, '{http://schemas.openxmlformats.org/presentationml/2006/main}tgtEl')
                    etree.SubElement(next_tgtEl, '{http://schemas.openxmlformats.org/presentationml/2006/main}sldTgt')

                    # Remove existing timing
                    for old_timing in slide_elem.findall('.//{http://schemas.openxmlformats.org/presentationml/2006/main}timing'):
                        slide_elem.remove(old_timing)

                    slide_elem.append(timing_elem)

                except Exception as e:
                    logger.warning(f"Failed to add bullet animations: {e}")

            # Apply color conversion
            primary_rgb = hex_to_rgb(theme["primary"])
            secondary_rgb = hex_to_rgb(theme["secondary"])
            accent_rgb = hex_to_rgb(theme["accent"])
            text_rgb = hex_to_rgb(theme["text"])
            light_gray_rgb = hex_to_rgb(theme["light_gray"])

            PRIMARY_COLOR = RGBColor(*primary_rgb)
            SECONDARY_COLOR = RGBColor(*secondary_rgb)
            ACCENT_COLOR = RGBColor(*accent_rgb)
            TEXT_COLOR = RGBColor(*text_rgb)
            LIGHT_GRAY = RGBColor(*light_gray_rgb)
            WHITE = RGBColor(0xFF, 0xFF, 0xFF)

            # Styling helper functions
            def apply_font_to_paragraph(p, font_name: str, is_heading: bool = False):
                # Apply fonts from template theme or config - don't skip when using template
                # The font_name comes from template_theme extraction when use_template_styling is True
                if font_name:
                    p.font.name = font_name

            def apply_color_to_paragraph(p, color, use_theme_color: bool = None):
                """Apply color to paragraph text.

                When use_template_styling is True and the color is TEXT_COLOR (black),
                we use the theme's text color scheme (tx1/dk1) for better theme integration.
                This ensures the color adapts if the template theme is changed.

                Args:
                    p: Paragraph object
                    color: RGBColor to apply
                    use_theme_color: Override to force theme color usage
                """
                if not color:
                    return

                # Determine if we should use theme colors
                should_use_theme = use_theme_color if use_theme_color is not None else use_template_styling

                # For text that would be black (#000000), use theme text color when template styling
                if should_use_theme and color == TEXT_COLOR:
                    try:
                        # Apply theme color via XML manipulation
                        # This uses schemeClr instead of srgbClr for proper theme integration
                        from pptx.oxml.ns import qn
                        from lxml import etree

                        # Get the run element (a:r) which contains the text properties
                        for run in p.runs:
                            rPr = run._r.get_or_add_rPr()
                            # Remove any existing solidFill
                            existing_fill = rPr.find(qn('a:solidFill'))
                            if existing_fill is not None:
                                rPr.remove(existing_fill)
                            # Add schemeClr for theme text color
                            solidFill = etree.SubElement(rPr, qn('a:solidFill'))
                            schemeClr = etree.SubElement(solidFill, qn('a:schemeClr'))
                            schemeClr.set('val', 'tx1')  # Theme text color 1 (dark)
                        return
                    except Exception as e:
                        logger.debug(f"Could not apply theme color, falling back to RGB: {e}")

                # Fallback to direct RGB color
                p.font.color.rgb = color

            # Get accent color from template theme for visual emphasis
            def get_template_accent_color(accent_num: int = 1) -> RGBColor:
                """Get accent color from template theme for selective visual emphasis.

                Args:
                    accent_num: 1-6 for different accent colors

                Returns:
                    RGBColor from template theme accent, or fallback to extracted primary
                """
                if use_template_styling and template_theme:
                    logger.debug(
                        "Accent color lookup",
                        accent_num=accent_num,
                        template_theme_keys=list(template_theme.keys()) if template_theme else [],
                    )
                    # Try direct accent key first (accent1, accent2, etc.)
                    accent_key = f"accent{accent_num}"
                    hex_color = template_theme.get(accent_key)

                    # Fallback to mapped keys (accent1 -> primary, accent2 -> accent1, etc.)
                    if not hex_color:
                        if accent_num == 1:
                            hex_color = template_theme.get("primary")
                        elif accent_num == 2:
                            hex_color = template_theme.get("accent1")
                        elif accent_num == 3:
                            hex_color = template_theme.get("accent2")

                    if hex_color:
                        try:
                            rgb = hex_to_rgb(hex_color)
                            logger.info(f"Using accent{accent_num} color: #{hex_color}")
                            return RGBColor(*rgb)
                        except Exception as e:
                            logger.warning(f"Failed to parse accent color: {e}")
                else:
                    logger.debug(
                        "Accent color skipped - no template styling",
                        use_template_styling=use_template_styling,
                        has_template_theme=bool(template_theme),
                    )
                return ACCENT_COLOR

            def apply_accent_to_run(run, accent_num: int = 1):
                """Apply accent color to a text run for visual emphasis."""
                accent_color = get_template_accent_color(accent_num)
                run.font.color.rgb = accent_color

            # Theme visual properties
            slide_background_style = theme.get("slide_background", "solid")
            header_style = theme.get("header_style", "none")
            bullet_style = theme.get("bullet_style", "circle")
            accent_position = theme.get("accent_position", "top")

            BULLET_CHARS = {
                "circle": ["•", "◦", "▪"],
                "circle-filled": ["●", "○", "▪"],
                "arrow": ["▸", "▹", "▫"],
                "chevron": ["»", "›", "·"],
                "dash": ["—", "-", "·"],
                "square": ["■", "□", "▪"],
                "number": None,
                "leaf": ["❧", "✿", "·"],
            }

            def get_bullet_chars():
                return BULLET_CHARS.get(bullet_style, BULLET_CHARS["circle"])

            def apply_slide_background(slide, is_title_slide=False):
                if use_template_styling:
                    return None

                bg_style = slide_background_style
                if bg_style in ("solid", "white"):
                    bg_shape = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        Inches(0), Inches(0),
                        prs.slide_width, prs.slide_height
                    )
                    if bg_style == "white":
                        bg_shape.fill.solid()
                        bg_shape.fill.fore_color.rgb = RGBColor(255, 255, 255)
                    else:
                        bg_shape.fill.solid()
                        bg_shape.fill.fore_color.rgb = PRIMARY_COLOR if is_title_slide else RGBColor(255, 255, 255)
                    bg_shape.line.fill.background()
                    return bg_shape
                elif bg_style in ("gradient", "warm-gradient"):
                    half_height = prs.slide_height / 2
                    top_shape = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        Inches(0), Inches(0),
                        prs.slide_width, half_height if not is_title_slide else prs.slide_height
                    )
                    top_shape.fill.solid()
                    top_shape.fill.fore_color.rgb = PRIMARY_COLOR if is_title_slide else SECONDARY_COLOR
                    top_shape.line.fill.background()
                    if not is_title_slide:
                        bottom_shape = slide.shapes.add_shape(
                            MSO_SHAPE.RECTANGLE,
                            Inches(0), half_height,
                            prs.slide_width, half_height
                        )
                        bottom_shape.fill.solid()
                        bottom_shape.fill.fore_color.rgb = RGBColor(250, 250, 250)
                        bottom_shape.line.fill.background()
                    return top_shape
                elif bg_style == "dark":
                    bg_shape = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        Inches(0), Inches(0),
                        prs.slide_width, prs.slide_height
                    )
                    bg_shape.fill.solid()
                    bg_shape.fill.fore_color.rgb = RGBColor(26, 26, 46)
                    bg_shape.line.fill.background()
                    return bg_shape
                else:
                    bg_shape = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        Inches(0), Inches(0),
                        prs.slide_width, prs.slide_height
                    )
                    bg_shape.fill.solid()
                    bg_shape.fill.fore_color.rgb = PRIMARY_COLOR if is_title_slide else RGBColor(255, 255, 255)
                    bg_shape.line.fill.background()
                    return bg_shape

            def add_header_accent(slide, title_text=""):
                if use_template_styling:
                    return None

                h_style = header_style
                if h_style in ("bar", "colorblock"):
                    bar = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        Inches(0), Inches(0),
                        prs.slide_width, Inches(1.2)
                    )
                    bar.fill.solid()
                    bar.fill.fore_color.rgb = PRIMARY_COLOR
                    bar.line.fill.background()
                    return bar
                elif h_style == "underline":
                    bar = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        Inches(0), Inches(0),
                        prs.slide_width, Inches(1.2)
                    )
                    bar.fill.solid()
                    bar.fill.fore_color.rgb = PRIMARY_COLOR
                    bar.line.fill.background()
                    underline = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        Inches(0.8), Inches(1.15),
                        Inches(3), Inches(0.05)
                    )
                    underline.fill.solid()
                    underline.fill.fore_color.rgb = SECONDARY_COLOR
                    underline.line.fill.background()
                    return bar
                elif h_style == "glow":
                    bar = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        Inches(0), Inches(0),
                        prs.slide_width, Inches(1.3)
                    )
                    bar.fill.solid()
                    bar.fill.fore_color.rgb = PRIMARY_COLOR
                    bar.line.fill.background()
                    glow_line = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        Inches(0), Inches(1.25),
                        prs.slide_width, Inches(0.08)
                    )
                    glow_line.fill.solid()
                    glow_line.fill.fore_color.rgb = ACCENT_COLOR
                    glow_line.line.fill.background()
                    return bar
                elif h_style == "none":
                    return None
                else:
                    bar = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        Inches(0), Inches(0),
                        prs.slide_width, Inches(1.2)
                    )
                    bar.fill.solid()
                    bar.fill.fore_color.rgb = PRIMARY_COLOR
                    bar.line.fill.background()
                    return bar

            def add_accent_elements(slide, position="top"):
                pos = accent_position
                if pos == "top":
                    accent = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        Inches(0), Inches(0),
                        prs.slide_width, Inches(0.08)
                    )
                    accent.fill.solid()
                    accent.fill.fore_color.rgb = SECONDARY_COLOR
                    accent.line.fill.background()
                elif pos == "bottom":
                    accent = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        Inches(0), prs.slide_height - Inches(0.08),
                        prs.slide_width, Inches(0.08)
                    )
                    accent.fill.solid()
                    accent.fill.fore_color.rgb = SECONDARY_COLOR
                    accent.line.fill.background()

            # Luminance and contrast functions
            def calculate_luminance(color: RGBColor) -> float:
                # RGBColor may use properties or indexing - handle both
                try:
                    # Try tuple-style indexing first
                    r = color[0] / 255.0
                    g = color[1] / 255.0
                    b = color[2] / 255.0
                except (TypeError, IndexError):
                    # Fall back to property access (r, g, b attributes)
                    r = getattr(color, 'r', getattr(color, 'red', 0)) / 255.0
                    g = getattr(color, 'g', getattr(color, 'green', 0)) / 255.0
                    b = getattr(color, 'b', getattr(color, 'blue', 0)) / 255.0
                r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
                g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
                b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
                return 0.2126 * r + 0.7152 * g + 0.0722 * b

            def get_contrast_ratio(color1: RGBColor, color2: RGBColor) -> float:
                lum1 = calculate_luminance(color1)
                lum2 = calculate_luminance(color2)
                lighter = max(lum1, lum2)
                darker = min(lum1, lum2)
                return (lighter + 0.05) / (darker + 0.05)

            def get_best_text_color(bg_color: RGBColor) -> RGBColor:
                white_contrast = get_contrast_ratio(bg_color, WHITE)
                black_contrast = get_contrast_ratio(bg_color, RGBColor(0, 0, 0))
                return WHITE if white_contrast >= black_contrast else RGBColor(0, 0, 0)

            def get_text_color_for_background(is_header=False, is_dark_bg=False, bg_color=None, is_content_slide=False):
                if slide_background_style == "dark":
                    return RGBColor(228, 228, 228)
                if bg_color is not None:
                    return get_best_text_color(bg_color)
                if slide_background_style in ("gradient", "warm-gradient") and is_content_slide:
                    return get_best_text_color(SECONDARY_COLOR)
                if is_header or is_dark_bg:
                    return get_best_text_color(PRIMARY_COLOR)
                return TEXT_COLOR

            # Determine include_sources from job metadata or config
            include_sources_override = job.metadata.get("include_sources")
            include_sources = include_sources_override if include_sources_override is not None else config.include_sources

            # ENTERPRISE FEATURE: Executive Summary slide
            # Provides key takeaways upfront (McKinsey/BCG standard)
            include_exec_summary = job.metadata.get("include_executive_summary", True)  # Default ON for enterprise

            # Calculate total slides
            total_slides = 1  # Title slide
            if config.include_toc:
                total_slides += 1
            if include_exec_summary:
                total_slides += 1  # Executive summary slide
            total_slides += len(job.sections)
            if include_sources and job.sources_used:
                total_slides += 1

            current_slide = 0

            # Image generation setup
            include_images = job.metadata.get("include_images", config.include_images)
            auto_charts_enabled = job.metadata.get("auto_charts", config.auto_charts)
            section_images = {}

            if include_images:
                try:
                    from backend.services.image_generator import ImageGeneratorConfig, ImageBackend, get_image_generator

                    image_config = ImageGeneratorConfig(
                        enabled=True,
                        backend=ImageBackend(config.image_backend),
                        default_width=400,
                        default_height=300,
                    )
                    image_service = get_image_generator(image_config)

                    sections_data = [
                        (section.title, section.revised_content or section.content)
                        for section in job.sections
                    ]
                    images = await image_service.generate_batch(
                        sections=sections_data,
                        document_title=job.title,
                    )

                    for idx, img in enumerate(images):
                        if img and img.success and img.path:
                            section_images[idx] = img.path

                except Exception as e:
                    logger.warning(f"Image generation failed: {e}")

            # Footer function
            def add_footer(slide, page_num, total_pages, include_branding: bool = True):
                """
                ENTERPRISE FOOTER: Add professional footer with page number and optional branding.

                When using template styling, tries to use template's footer placeholders first.
                Falls back to manual text boxes if placeholders don't exist.

                Consulting standard footers include:
                - Company/document identifier (left)
                - Confidentiality notice (optional, center)
                - Page numbers (right)

                Args:
                    slide: The slide to add footer to
                    page_num: Current page number
                    total_pages: Total page count
                    include_branding: Whether to include company name/date
                """
                from pptx.enum.shapes import PP_PLACEHOLDER

                # Position footer so bottom edge stays within slide bounds
                footer_height = Inches(0.25)
                footer_y = prs.slide_height - footer_height - Inches(0.02)  # 0.02" margin from bottom
                logger.debug(f"Footer position: y={footer_y / 914400:.2f}in (slide height={prs.slide_height / 914400:.2f}in)")
                footer_font_size = Pt(9)  # Enterprise standard: 8-10pt for footers

                # Try to use template placeholders when template styling is enabled
                used_placeholder = False
                if use_template_styling:
                    try:
                        for shape in slide.placeholders:
                            ph_type = shape.placeholder_format.type
                            if ph_type == PP_PLACEHOLDER.SLIDE_NUMBER:
                                shape.text = f"{page_num} / {total_pages}"
                                used_placeholder = True
                            elif ph_type == PP_PLACEHOLDER.FOOTER:
                                company_name = job.metadata.get("company_name", "")
                                generation_date = datetime.now().strftime("%b %Y")
                                if company_name:
                                    shape.text = f"{company_name} | {generation_date}"
                                else:
                                    shape.text = generation_date
                                used_placeholder = True
                            elif ph_type == PP_PLACEHOLDER.DATE:
                                shape.text = datetime.now().strftime("%B %d, %Y")
                                used_placeholder = True
                    except Exception as ph_err:
                        logger.debug(f"Could not use footer placeholders: {ph_err}")

                # Fall back to manual footer if placeholders not available
                if not used_placeholder:
                    # RIGHT: Page number (always present)
                    page_footer = slide.shapes.add_textbox(
                        prs.slide_width - Inches(1.2), footer_y,
                        Inches(1.0), footer_height
                    )
                    tf = page_footer.text_frame
                    tf.paragraphs[0].text = f"{page_num} / {total_pages}"
                    tf.paragraphs[0].font.size = footer_font_size
                    tf.paragraphs[0].font.color.rgb = LIGHT_GRAY
                    tf.paragraphs[0].alignment = PP_ALIGN.RIGHT

                    # LEFT: Company name and date (enterprise branding)
                    if include_branding:
                        # Get company name from job metadata or use default
                        company_name = job.metadata.get("company_name", "")
                        generation_date = datetime.now().strftime("%b %Y")

                        if company_name:
                            brand_text = f"{company_name} | {generation_date}"
                        else:
                            brand_text = generation_date

                        brand_footer = slide.shapes.add_textbox(
                            Inches(0.5), footer_y,
                            Inches(3.0), footer_height
                        )
                        tf = brand_footer.text_frame
                        tf.paragraphs[0].text = brand_text
                        tf.paragraphs[0].font.size = footer_font_size
                        tf.paragraphs[0].font.color.rgb = LIGHT_GRAY
                        tf.paragraphs[0].alignment = PP_ALIGN.LEFT

                # CENTER: Confidentiality notice (optional, for enterprise docs)
                confidentiality = job.metadata.get("confidentiality_notice", "")
                if confidentiality:
                    conf_footer = slide.shapes.add_textbox(
                        Inches(4.0), footer_y,
                        Inches(5.0), footer_height
                    )
                    tf = conf_footer.text_frame
                    tf.paragraphs[0].text = confidentiality
                    tf.paragraphs[0].font.size = Pt(8)
                    tf.paragraphs[0].font.color.rgb = LIGHT_GRAY
                    tf.paragraphs[0].alignment = PP_ALIGN.CENTER

            # ========== TITLE SLIDE ==========
            current_slide += 1
            title_layout = get_slide_layout("title")
            slide = prs.slides.add_slide(title_layout)
            # On title slide, preserve picture placeholders which may contain background images/branding
            # Only remove empty text placeholders, not visual elements
            remove_empty_placeholders(slide, preserve_branding=True, keep_picture_placeholder=True)

            # Copy embedded branding elements (logos, background images) from layout to slide
            # python-pptx doesn't automatically copy non-placeholder shapes from layouts
            if use_template_styling and title_layout:
                copy_layout_branding_to_slide(slide, title_layout)

            if enable_animations:
                add_slide_transition(slide, "fade", duration=transition_duration, speed=animation_speed)

            if not use_template_styling:
                apply_slide_background(slide, is_title_slide=True)
                accent_bar_height = prs.slide_height * 0.133
                accent_bar = slide.shapes.add_shape(
                    MSO_SHAPE.RECTANGLE,
                    Inches(0), prs.slide_height - accent_bar_height,
                    prs.slide_width, accent_bar_height
                )
                accent_bar.fill.solid()
                accent_bar.fill.fore_color.rgb = SECONDARY_COLOR
                accent_bar.line.fill.background()

            title_text_color = get_text_color_for_background(is_dark_bg=True)

            content_width = prs.slide_width - Inches(1.6)
            title_box = slide.shapes.add_textbox(
                scaled_inches(0.8, "w"), scaled_inches(2.5, "h"),
                content_width, scaled_inches(1.5, "h")
            )
            tf = title_box.text_frame
            tf.word_wrap = True
            tf.auto_size = MSO_AUTO_SIZE.NONE
            p = tf.paragraphs[0]
            title_text = sanitize_text(job.title) or "Untitled"
            if len(title_text) > 80:
                # Use LLM to intelligently rewrite title instead of truncating with "..."
                title_text = await llm_rewrite_for_slide(title_text, 80, text_type="title")
            p.text = title_text
            apply_font_to_paragraph(p, heading_font, is_heading=True)
            p.font.size = Pt(48)
            p.font.bold = True
            apply_color_to_paragraph(p, title_text_color)
            # Add line spacing for multi-line titles to prevent overlap with background elements
            p.line_spacing = 1.2  # 120% line spacing
            p.space_after = Pt(4)

            # Subtitle
            desc_box = slide.shapes.add_textbox(
                scaled_inches(0.8, "w"), scaled_inches(4.2, "h"),
                content_width, scaled_inches(1, "h")
            )
            tf = desc_box.text_frame
            tf.word_wrap = True
            tf.auto_size = MSO_AUTO_SIZE.NONE
            p = tf.paragraphs[0]
            desc_text = getattr(job.outline, 'description', None) if job.outline else None
            desc_text = desc_text or job.description
            desc_text = sanitize_text(desc_text) or ""
            if len(desc_text) > 200:
                # Use LLM to intelligently condense description
                desc_text = await llm_rewrite_for_slide(desc_text, 200, text_type="bullet")
            p.text = desc_text
            apply_font_to_paragraph(p, body_font)
            p.font.size = Pt(20)
            apply_color_to_paragraph(p, ACCENT_COLOR)
            # Add line spacing for multi-line subtitles
            p.line_spacing = 1.2  # 120% line spacing
            p.space_after = Pt(4)

            # Target audience subtitle (if available from outline)
            target_audience = getattr(job.outline, 'target_audience', None) if job.outline else None
            if target_audience:
                audience_box = slide.shapes.add_textbox(
                    scaled_inches(0.8, "w"), scaled_inches(5.0, "h"),
                    content_width, scaled_inches(0.5, "h")
                )
                tf = audience_box.text_frame
                tf.word_wrap = True
                tf.auto_size = MSO_AUTO_SIZE.NONE
                p = tf.paragraphs[0]
                p.text = sanitize_text(f"Prepared for: {target_audience}")
                apply_font_to_paragraph(p, body_font)
                p.font.size = Pt(14)
                p.font.italic = True
                apply_color_to_paragraph(p, title_text_color)

            # Date
            generation_date = datetime.now()
            date_box = slide.shapes.add_textbox(
                scaled_inches(0.8, "w"), prs.slide_height - Inches(0.8),
                scaled_inches(4, "w"), Inches(0.4)
            )
            tf = date_box.text_frame
            p = tf.paragraphs[0]
            p.text = generation_date.strftime("%B %d, %Y")
            apply_font_to_paragraph(p, body_font)
            p.font.size = Pt(14)
            p.font.color.rgb = title_text_color

            # Title slide notes - PHASE 10: Include template filename
            output_language = job.metadata.get("output_language", "en")
            user_info = job.metadata.get("user_email") or job.user_id
            from backend.services.llm import llm_config as _llm_cfg
            llm_model = job.metadata.get("llm_model") or config.model or _llm_cfg.default_chat_model
            template_filename = job.metadata.get("template_pptx_filename", "")

            title_notes = f"""Generated by AIDocumentIndexer

Model: {llm_model}
User: {user_info}
Date: {generation_date.strftime('%Y-%m-%d %H:%M')}
Language: {output_language}
Theme: {theme_key}
Font: {font_family_key}
Total sections: {len(job.sections)}
"""
            # Add template source if a template file was used
            if template_filename:
                title_notes += f"Template Source: {template_filename}\n"

            add_slide_notes(slide, title_notes)

            # TOC tracking
            toc_paragraphs = []
            toc_section_indices = []
            content_text_color = TEXT_COLOR

            # ========== TABLE OF CONTENTS SLIDE ==========
            if config.include_toc:
                current_slide += 1
                toc_layout = get_slide_layout("toc")
                slide = prs.slides.add_slide(toc_layout)
                remove_empty_placeholders(slide)

                # Copy branding elements (logos, backgrounds) from layout to slide
                if use_template_styling:
                    copy_layout_branding_to_slide(slide, toc_layout)

                if enable_animations:
                    add_slide_transition(slide, "push", duration=transition_duration, speed=animation_speed)

                if not use_template_styling:
                    add_header_accent(slide, "Contents")

                title_y = scaled_inches(0.3, "h") if header_style != "none" else scaled_inches(0.5, "h")
                if use_template_styling:
                    title_color = TEXT_COLOR
                else:
                    title_color = WHITE if header_style != "none" else TEXT_COLOR

                # Calculate safe title width respecting logo zone (same logic as content slides)
                toc_title_width = prs.slide_width - Inches(3.0)  # Default: leave 3" right margin for logo
                if slide_content_planner and hasattr(slide_content_planner.learner, 'detected_logo_x'):
                    logo_x = slide_content_planner.learner.detected_logo_x
                    if logo_x is not None:
                        toc_title_width = logo_x - scaled_inches(0.6, "w") - Inches(0.3)
                toc_title_width = max(Inches(4.0), min(toc_title_width, prs.slide_width - Inches(3.0)))

                toc_title = slide.shapes.add_textbox(
                    scaled_inches(0.6, "w"), title_y,
                    toc_title_width, scaled_inches(0.8, "h")
                )
                tf = toc_title.text_frame
                tf.word_wrap = True
                tf.auto_size = MSO_AUTO_SIZE.NONE
                p = tf.paragraphs[0]
                p.text = "Contents"
                apply_font_to_paragraph(p, heading_font, is_heading=True)
                p.font.size = Pt(36)
                p.font.bold = True
                apply_color_to_paragraph(p, title_color)

                if header_style == "none" and not use_template_styling:
                    add_accent_elements(slide)

                toc_bullets = get_bullet_chars()
                toc_box = slide.shapes.add_textbox(
                    scaled_inches(0.6, "w"), scaled_inches(2.0, "h"),
                    prs.slide_width - Inches(1.2), scaled_inches(4.8, "h")
                )
                tf = toc_box.text_frame
                tf.word_wrap = True
                tf.auto_size = MSO_AUTO_SIZE.NONE

                if use_template_styling:
                    content_text_color = TEXT_COLOR
                else:
                    content_text_color = get_text_color_for_background(is_content_slide=True)

                # Add accent underline below TOC title for visual separation
                if use_template_styling:
                    try:
                        toc_accent_color = get_template_accent_color(1)
                        toc_underline = slide.shapes.add_shape(
                            MSO_SHAPE.RECTANGLE,
                            scaled_inches(0.6, "w"), scaled_inches(1.5, "h"),
                            scaled_inches(3.0, "w"), Inches(0.03)
                        )
                        toc_underline.fill.solid()
                        toc_underline.fill.fore_color.rgb = toc_accent_color
                        toc_underline.line.fill.background()
                    except Exception as toc_accent_err:
                        logger.debug(f"Could not add TOC accent underline: {toc_accent_err}")

                first_toc_used = False
                for idx, section in enumerate(job.sections):
                    if first_toc_used:
                        p = tf.add_paragraph()
                    else:
                        p = tf.paragraphs[0]
                        first_toc_used = True
                    section_title = sanitize_text(section.title) or f"Section {idx + 1}"
                    if len(section_title) > 70:
                        # Use LLM to shorten TOC section title
                        section_title = await llm_rewrite_for_slide(section_title, 70, text_type="title")

                    roman_pattern = r'^[IVXLCDM]+\.\s+'
                    if re.match(roman_pattern, section_title):
                        section_title = re.sub(roman_pattern, '', section_title)

                    # Use runs to apply accent color to numbers for visual hierarchy
                    p.clear()
                    number_run = p.add_run()
                    number_run.text = f"{idx + 1}."
                    number_run.font.size = Pt(20)
                    number_run.font.bold = True
                    if use_template_styling:
                        # Apply accent color to numbers when using template
                        apply_accent_to_run(number_run, accent_num=1)
                    else:
                        apply_font_to_paragraph(p, body_font)
                        number_run.font.color.rgb = ACCENT_COLOR

                    title_run = p.add_run()
                    title_run.text = f"  {section_title}"
                    title_run.font.size = Pt(20)
                    title_run.font.bold = True
                    if not use_template_styling:
                        title_run.font.color.rgb = content_text_color
                        if body_font:
                            title_run.font.name = body_font

                    p.space_after = Pt(4)

                    # Add section description if available (from outline generation)
                    desc = section.description if hasattr(section, 'description') and section.description else None
                    if desc and len(desc.strip()) > 10:
                        desc_p = tf.add_paragraph()
                        desc_text = sanitize_text(desc)[:100]
                        desc_run = desc_p.add_run()
                        desc_run.text = f"     {desc_text}"
                        desc_run.font.size = Pt(12)
                        desc_run.font.italic = True
                        if use_template_styling:
                            desc_run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
                        else:
                            desc_run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
                        desc_p.space_after = Pt(10)
                    else:
                        p.space_after = Pt(12)

                    toc_paragraphs.append(p)
                    toc_section_indices.append(idx)

                if not first_toc_used:
                    p = tf.paragraphs[0]
                    p.text = "No sections"
                    apply_font_to_paragraph(p, body_font)
                    p.font.size = Pt(20)

                if enable_animations and first_toc_used:
                    add_bullet_animations(slide, toc_box, animation_type="appear", delay_between=300)

                add_footer(slide, current_slide, total_slides)

            # ========== EXECUTIVE SUMMARY SLIDE ==========
            # ENTERPRISE STANDARD: Key takeaways upfront (McKinsey/BCG pyramid principle)
            if include_exec_summary and job.sections:
                current_slide += 1
                exec_layout = get_slide_layout("content")
                slide = prs.slides.add_slide(exec_layout)
                remove_empty_placeholders(slide)

                # Copy branding elements (logos, backgrounds) from layout to slide
                if use_template_styling:
                    copy_layout_branding_to_slide(slide, exec_layout)

                if enable_animations:
                    add_slide_transition(slide, "fade", duration=transition_duration, speed=animation_speed)

                if not use_template_styling:
                    add_header_accent(slide, "Executive Summary")

                # Title
                title_y = scaled_inches(0.3, "h") if header_style != "none" else scaled_inches(0.5, "h")
                if use_template_styling:
                    exec_title_color = TEXT_COLOR
                else:
                    exec_title_color = WHITE if header_style != "none" else TEXT_COLOR

                exec_title_box = slide.shapes.add_textbox(
                    scaled_inches(0.6, "w"), title_y,
                    prs.slide_width - Inches(3.0), scaled_inches(0.8, "h")  # Leave room for logo
                )
                tf = exec_title_box.text_frame
                tf.word_wrap = True
                tf.auto_size = MSO_AUTO_SIZE.NONE
                p = tf.paragraphs[0]
                p.text = "Executive Summary"
                apply_font_to_paragraph(p, heading_font, is_heading=True)
                p.font.size = Pt(36)
                p.font.bold = True
                apply_color_to_paragraph(p, exec_title_color)

                if header_style == "none" and not use_template_styling:
                    add_accent_elements(slide)

                # Generate executive summary using LLM
                try:
                    all_section_content = "\n\n".join([
                        f"Section: {s.title}\n{s.revised_content or s.content}"
                        for s in job.sections
                    ])

                    from backend.services.llm import EnhancedLLMFactory
                    llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                        operation="content_generation",
                        prefer_fast=True,
                    )

                    exec_summary_prompt = f"""Create an EXECUTIVE SUMMARY for a presentation about: {job.title}

Based on this content:
{all_section_content[:8000]}

Requirements (McKinsey/BCG standard):
1. Write 3-5 KEY TAKEAWAYS - each a COMPLETE sentence/thought
2. Each bullet must be an ACTION statement (what decision/action follows)
3. Lead with the most important insight first (pyramid principle)
4. Maximum 65 characters per bullet (HARD LIMIT - shorter is better)
5. Use specific numbers/metrics when available
6. Focus on "so what?" - implications for the audience

CRITICAL: Every bullet MUST be a COMPLETE THOUGHT. Never end with:
- Conjunctions (and, or, but)
- Prepositions (to, for, of, with, in, on, at, by)
- Articles (the, a, an)
- Incomplete verbs (is, are, will, can, should)
- Numbers without context (e.g., "for 3" without saying "for 3 years")

Format: Return ONLY the bullets, one per line, starting with "•"

Example GOOD takeaways (complete thoughts):
• Revenue grew 23% YoY driven by APAC expansion
• Customer retention reached 94% last quarter
• Recommend 15% marketing spend increase in Q2
• Launch social media campaign targeting millennials
• Partner with 3 major influencers for brand awareness

Example BAD takeaways (NEVER write these):
• Launch targeted content campaigns highlighting the brand's values and (INCOMPLETE - ends with "and")
• Collaborate with influencers to amplify (INCOMPLETE - ends with "amplify")
• Partner with FC Bayern Munich for 3 (INCOMPLETE - ends with number)
• Utilize analytics to monitor performance and make (INCOMPLETE - ends with "make")
"""

                    exec_summary_result = await llm.ainvoke(exec_summary_prompt)
                    # Safely extract content, handling None case
                    exec_summary_text = None
                    if hasattr(exec_summary_result, 'content') and exec_summary_result.content:
                        exec_summary_text = exec_summary_result.content
                    if not exec_summary_text:
                        exec_summary_text = str(exec_summary_result) if exec_summary_result else ""

                    # Parse and validate bullets - ensure complete thoughts
                    exec_bullets = []
                    for line in (exec_summary_text or "").strip().split('\n'):
                        line = line.strip()
                        if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                            # Remove bullet prefix and clean
                            bullet_text = re.sub(r'^[•\-\*]\s*', '', line).strip()
                            # Strip any markdown formatting (bold, italic, etc.)
                            bullet_text = strip_markdown(bullet_text)
                            if bullet_text and len(bullet_text) > 10:  # Skip trivial bullets
                                # Use ensure_complete_thought for smart truncation
                                if len(bullet_text) > 70:
                                    bullet_text = ensure_complete_thought(bullet_text, 70)
                                # Final validation: skip if still incomplete
                                if is_sentence_complete(bullet_text):
                                    exec_bullets.append(bullet_text)
                                else:
                                    # Log warning but still include if above minimum length
                                    logger.warning(f"Executive summary bullet may be incomplete: {bullet_text[:40]}...")
                                    if len(bullet_text) >= 25:  # Only include if reasonably long
                                        exec_bullets.append(bullet_text)

                    # Limit to 5 key takeaways
                    exec_bullets = exec_bullets[:5]

                    # Detect generic placeholder bullets from LLM (e.g., "Key Takeaway", "Action")
                    # Small models sometimes return template text instead of real content
                    if exec_bullets:
                        generic_phrases = {"key takeaway", "action", "key insight", "main point", "takeaway", "action item"}
                        all_generic = all(
                            any(g in b.lower() for g in generic_phrases) for b in exec_bullets
                        )
                        if all_generic:
                            logger.warning("Executive summary bullets are generic placeholders, falling back to section titles")
                            exec_bullets = [sanitize_text(s.title)[:65] for s in job.sections[:5] if s.title]

                except Exception as e:
                    logger.warning(f"Could not generate executive summary via LLM: {e}")
                    # Fallback: extract key points from section titles
                    exec_bullets = []
                    for s in job.sections[:5]:
                        title = sanitize_text(s.title)[:65]
                        if title:
                            exec_bullets.append(title)

                # Add executive summary content box
                exec_content_box = slide.shapes.add_textbox(
                    scaled_inches(0.6, "w"), scaled_inches(1.8, "h"),
                    prs.slide_width - Inches(1.2), scaled_inches(5.0, "h")
                )
                tf = exec_content_box.text_frame
                tf.word_wrap = True
                tf.auto_size = MSO_AUTO_SIZE.NONE

                # Set vertical alignment for executive summary
                set_vertical_alignment(tf, 'top')

                if use_template_styling:
                    exec_content_color = TEXT_COLOR
                else:
                    exec_content_color = get_text_color_for_background(is_content_slide=True)

                exec_bullet_chars = get_bullet_chars()
                first_exec_bullet = True

                for idx, bullet in enumerate(exec_bullets):
                    if first_exec_bullet:
                        p = tf.paragraphs[0]
                        first_exec_bullet = False
                    else:
                        p = tf.add_paragraph()

                    p.level = 0
                    bullet_char = exec_bullet_chars[0] if exec_bullet_chars else "•"
                    p.text = f"{bullet_char} {bullet}"
                    apply_font_to_paragraph(p, body_font)
                    p.font.size = Pt(22)  # Slightly larger for emphasis
                    p.font.bold = False  # Body text should not be bold - only titles
                    apply_color_to_paragraph(p, exec_content_color)
                    p.space_after = Pt(18)  # More spacing for readability

                if not exec_bullets:
                    p = tf.paragraphs[0]
                    p.text = "• Key insights from the presentation"
                    apply_font_to_paragraph(p, body_font)
                    p.font.size = Pt(22)

                if enable_animations:
                    add_bullet_animations(slide, exec_content_box, animation_type="fade", delay_between=400)

                # Add speaker notes for executive summary
                exec_notes = f"""EXECUTIVE SUMMARY SLIDE

Key Takeaways:
{chr(10).join(['- ' + b for b in exec_bullets])}

Presenter guidance:
- This slide captures the "so what?" of the entire presentation
- Spend 60-90 seconds on this slide
- Each bullet should connect to a later detailed slide
- Use this to set expectations and frame the narrative
"""
                add_slide_notes(slide, exec_notes)

                add_footer(slide, current_slide, total_slides)

            # ========== CONTENT SLIDES ==========
            content_transitions = ["wipe", "fade", "push", "dissolve"]
            content_bullets = get_bullet_chars()
            content_slides_by_section = {}

            for section_idx, section in enumerate(job.sections):
                current_slide += 1
                has_image = section_idx in section_images

                # ========== SECTION VALIDATION ==========
                # Validate section has required content before processing
                if not section.title or not section.title.strip():
                    logger.warning(
                        f"Section {section_idx + 1} has empty title, using default",
                        section_idx=section_idx,
                    )
                    section.title = f"Section {section_idx + 1}"

                if not section.content or not section.content.strip():
                    logger.warning(
                        f"Section {section_idx + 1} has empty content, adding placeholder",
                        section_idx=section_idx,
                        title=section.title[:50] if section.title else "No title",
                    )
                    section.content = "• Content not available for this section"

                # ========== CONTENT ANALYSIS FOR LAYOUT RECOMMENDATIONS ==========
                content_for_analysis = (section.revised_content or section.content or "").lower()
                if content_for_analysis:
                    has_comparison = any(w in content_for_analysis for w in ['vs ', 'versus ', 'compared to ', 'difference between '])
                    has_process = any(w in content_for_analysis for w in ['step 1', 'step 2', 'first,', 'then,', 'finally,', 'phase 1', 'phase 2'])
                    has_statistics = bool(re.search(r'\d+%|\$[\d,.]+|\d+\.\d+[xX]', content_for_analysis))
                    if has_comparison:
                        logger.info(
                            "Content analysis: comparison detected, two-column layout recommended",
                            section=section.title[:50],
                            section_idx=section_idx + 1,
                        )
                    if has_process:
                        logger.info(
                            "Content analysis: process/steps detected, timeline layout recommended",
                            section=section.title[:50],
                            section_idx=section_idx + 1,
                        )
                    if has_statistics:
                        logger.info(
                            "Content analysis: statistics detected, chart visualization recommended",
                            section=section.title[:50],
                            section_idx=section_idx + 1,
                        )

                # ========== PRE-RENDER REVIEW & FIX (Optional) ==========
                if slide_reviewer:
                    try:
                        # Get layout-specific constraints from template analysis
                        layout_key = getattr(section, 'layout', None) or "standard"
                        layout_info = None
                        title_placeholder = None
                        body_placeholder = None

                        if template_analysis and hasattr(template_analysis, 'layouts'):
                            # Find matching layout from template
                            for layout in template_analysis.layouts:
                                if layout.layout_type == layout_key or layout.name.lower().replace(' ', '_') == layout_key:
                                    layout_info = layout
                                    break
                            # If no match, use first content layout
                            if not layout_info and template_analysis.layouts:
                                for layout in template_analysis.layouts:
                                    if 'content' in layout.layout_type.lower():
                                        layout_info = layout
                                        break

                            # Extract placeholder specs from layout
                            if layout_info:
                                for ph in layout_info.placeholders:
                                    if ph.type in ['title', 'ctrTitle']:
                                        title_placeholder = ph
                                    elif ph.type == 'body':
                                        body_placeholder = ph

                        # Build slide specification for review (including theme and layout info)
                        slide_spec = {
                            "title": section.title or f"Section {section_idx + 1}",
                            "bullets": slide_reviewer._extract_bullets(section.content or ""),
                            "layout": layout_key,
                            "has_image": has_image,
                            "speaker_notes": section.speaker_notes or "",
                            "slide_width": prs.slide_width.inches,
                            "slide_height": prs.slide_height.inches,
                            # Theme information for LLM review
                            "template_name": job.metadata.get("template_pptx_filename", "Default"),
                            "accent_color": template_theme.get("accent1", "#4F81BD") if template_theme else "#4F81BD",
                            "text_color": template_theme.get("text", "#000000") if template_theme else "#000000",
                            "primary_color": template_theme.get("primary", "#333333") if template_theme else "#333333",
                            "font_heading": template_theme.get("font_heading", "Calibri") if template_theme else "Calibri",
                            "font_body": template_theme.get("font_body", "Calibri") if template_theme else "Calibri",
                            "theme_style": theme.get("style_description", "") if theme else "",
                            # Layout-specific placeholder info
                            "title_placeholder": {
                                "width": title_placeholder.width_inches if title_placeholder else 12.0,
                                "height": title_placeholder.height_inches if title_placeholder else 1.0,
                                "x": title_placeholder.x_inches if title_placeholder else 0.5,
                                "y": title_placeholder.y_inches if title_placeholder else 0.5,
                                "font_size": title_placeholder.font_size_pt if title_placeholder else 32,
                                "max_chars": title_placeholder.max_chars if title_placeholder else 50,
                            },
                            "body_placeholder": {
                                "width": body_placeholder.width_inches if body_placeholder else 11.5,
                                "height": body_placeholder.height_inches if body_placeholder else 4.8,
                                "x": body_placeholder.x_inches if body_placeholder else 0.5,
                                "y": body_placeholder.y_inches if body_placeholder else 1.8,
                                "font_size": body_placeholder.font_size_pt if body_placeholder else 18,
                                "max_chars": body_placeholder.max_chars if body_placeholder else 500,
                                "max_bullets": body_placeholder.max_bullets if body_placeholder else 7,
                            },
                            "layout_recommended_use": layout_info.recommended_use if layout_info else "general content",
                        }

                        # Get template constraints - use learned layouts if available, otherwise from placeholders
                        if slide_content_planner:
                            # Use per-slide constraints from template learning
                            slide_plan = slide_content_planner.plan_slide_content(
                                section_order=section_idx,
                                total_sections=len(job.sections),
                                has_image=has_image,
                                section_title=section.title or "",
                            )
                            learned_layout = slide_plan.layout
                            template_constraints = {
                                "title_max_chars": slide_plan.title_constraints.get('max_chars', 50),
                                "bullet_max_chars": slide_plan.content_constraints.get('max_bullet_chars', 70),
                                "bullets_per_slide": slide_plan.content_constraints.get('max_bullets', 7 if has_image else 12),
                                "body_max_chars": 500,
                            }
                            logger.debug(
                                "Using learned layout constraints",
                                slide=section_idx + 1,
                                layout_type=learned_layout.layout_type,
                                max_title=template_constraints["title_max_chars"],
                                max_bullets=template_constraints["bullets_per_slide"],
                                max_bullet_chars=template_constraints["bullet_max_chars"],
                            )
                        else:
                            # Fallback to layout placeholders or defaults
                            template_constraints = {
                                "title_max_chars": title_placeholder.max_chars if title_placeholder else 50,
                                "bullet_max_chars": int((body_placeholder.width_inches if body_placeholder else 11.5) * 8),  # ~8 chars per inch at 18pt
                                "bullets_per_slide": body_placeholder.max_bullets if body_placeholder else (7 if has_image else 12),
                                "body_max_chars": body_placeholder.max_chars if body_placeholder else 500,
                            }

                        # Adjust for image presence (content area is narrower)
                        if has_image:
                            template_constraints["bullet_max_chars"] = min(template_constraints["bullet_max_chars"], 50)
                            template_constraints["bullets_per_slide"] = min(template_constraints["bullets_per_slide"], 7)

                        # HARD ENFORCEMENT: Truncate content BEFORE LLM review as failsafe
                        # This ensures slides never overflow regardless of what LLM produces
                        section.title = enforce_text_length(
                            section.title,
                            template_constraints["title_max_chars"],
                            "title"
                        )
                        section.content = enforce_bullet_constraints(
                            section.content,
                            max_bullet_chars=template_constraints["bullet_max_chars"],
                            max_bullets=template_constraints["bullets_per_slide"],
                        )

                        # Review and fix the slide content - already in async context
                        review_result, fix_result = await slide_reviewer.review_and_fix_slide(
                            slide_spec, section_idx, template_constraints
                        )

                        # Apply fixes to the section if any were made
                        if fix_result.fixes_applied > 0:
                            fixed = fix_result.fixed_content
                            if fixed.get("title") and fixed["title"] != section.title:
                                section.title = fixed["title"]
                                logger.info(f"Slide {section_idx + 1}: Fixed title")

                            # Reconstruct content from fixed bullets
                            if fixed.get("bullets"):
                                fixed_content_lines = []
                                for bullet in fixed["bullets"]:
                                    text = bullet.get("text", bullet) if isinstance(bullet, dict) else str(bullet)
                                    fixed_content_lines.append(f"• {text}")
                                    for sub in bullet.get("sub_bullets", []):
                                        sub_text = sub.get("text", sub) if isinstance(sub, dict) else str(sub)
                                        fixed_content_lines.append(f"  ◦ {sub_text}")
                                if fixed_content_lines:
                                    section.content = "\n".join(fixed_content_lines)
                                    logger.info(f"Slide {section_idx + 1}: Fixed bullets")

                            logger.info(
                                "Applied pre-render fixes",
                                slide=section_idx + 1,
                                fixes=fix_result.fixes_applied,
                                changes=fix_result.changes_made,
                            )
                        elif review_result.has_issues:
                            # Log issues that couldn't be auto-fixed
                            for issue in review_result.issues:
                                logger.warning(
                                    f"Slide {section_idx + 1}: [{issue.severity}] {issue.issue_type}: {issue.description}"
                                )
                    except Exception as review_err:
                        logger.warning(f"Pre-render review failed for slide {section_idx + 1}: {review_err}")

                # Use appropriate layout based on whether we have an image
                if has_image and use_template_styling:
                    content_layout = get_slide_layout("content", has_image=True)
                    slide = prs.slides.add_slide(content_layout)
                    # Keep picture placeholder for image
                    remove_empty_placeholders(slide, keep_picture_placeholder=True)
                else:
                    content_layout = get_slide_layout("content", has_image=False)
                    slide = prs.slides.add_slide(content_layout)
                    remove_empty_placeholders(slide, keep_picture_placeholder=False)

                # Copy branding elements (logos, backgrounds) from layout to slide
                if use_template_styling:
                    copy_layout_branding_to_slide(slide, content_layout)

                content_slides_by_section[section_idx] = slide

                if enable_animations:
                    trans_type = content_transitions[section_idx % len(content_transitions)]
                    add_slide_transition(slide, trans_type, duration=transition_duration, speed=animation_speed)

                if not use_template_styling:
                    add_header_accent(slide, section.title)

                # Section title
                title_y = scaled_inches(0.3, "h") if header_style != "none" else scaled_inches(0.5, "h")
                if use_template_styling:
                    title_color = TEXT_COLOR
                else:
                    title_color = WHITE if header_style != "none" else TEXT_COLOR

                # Get per-slide constraints if available
                # ALWAYS use planner constraints when available (regardless of reviewer status)
                max_title_chars_for_slide = 50  # Default
                max_bullet_chars_for_slide = 120  # PHASE 11: Increased from 70
                max_bullets_for_slide = 7 if has_image else 12  # Default

                if slide_content_planner:
                    # Use learned layout constraints from template analysis
                    slide_plan = slide_content_planner.plan_slide_content(
                        section_order=section_idx,
                        total_sections=len(job.sections),
                        has_image=has_image,
                        section_title=section.title or "",
                    )
                    max_title_chars_for_slide = slide_plan.title_constraints.get('max_chars', 50)
                    max_bullet_chars_for_slide = slide_plan.content_constraints.get('max_bullet_chars', 120)
                    max_bullets_for_slide = slide_plan.content_constraints.get('max_bullets', 7 if has_image else 12)

                    # Apply visual style adjustments from vision analysis
                    visual_style = getattr(slide_plan, 'visual_style', {}) or {}
                    if visual_style:
                        # Adjust font size based on content density
                        density = visual_style.get('density', 'normal')
                        if density == 'sparse':
                            # Increase font sizes for sparse content slides
                            base_font_size = Pt(24)
                            line_spacing = Pt(14)
                            logger.debug(f"Slide {section_idx + 1}: sparse layout, using larger fonts")
                        elif density == 'dense':
                            # Use smaller fonts for dense content
                            base_font_size = Pt(18)
                            line_spacing = Pt(8)
                            logger.debug(f"Slide {section_idx + 1}: dense layout, using smaller fonts")
                        # Note: 'normal' uses defaults set elsewhere

                    logger.debug(
                        f"Using template-learned constraints for slide {section_idx + 1}",
                        max_title=max_title_chars_for_slide,
                        max_bullet=max_bullet_chars_for_slide,
                        max_bullets=max_bullets_for_slide,
                        visual_style=visual_style.get('density') if visual_style else None,
                    )

                # Calculate safe title width respecting logo position
                # Default: leave 3" right margin for potential logo
                safe_title_width = prs.slide_width - Inches(3.0)
                title_left_pos = scaled_inches(0.6, "w")

                # Use detected logo position if available from template learner
                if slide_content_planner and hasattr(slide_content_planner.learner, 'detected_logo_x'):
                    logo_x = slide_content_planner.learner.detected_logo_x
                    if logo_x is not None:
                        # Leave 0.3" margin before logo
                        safe_title_width = logo_x - title_left_pos - Inches(0.3)
                        logger.debug(
                            f"Limiting title width to {safe_title_width / 914400:.2f}in based on logo at x={logo_x / 914400:.2f}in"
                        )

                # Ensure title width is at least 4" but no more than slide_width - 3"
                safe_title_width = max(Inches(4.0), min(safe_title_width, prs.slide_width - Inches(3.0)))

                # Get title text FIRST to calculate dynamic height
                title_text = sanitize_text(section.title) or f"Section {section_idx + 1}"
                # Use learned constraint or default 50 chars to prevent overlap with branding
                if len(title_text) > max_title_chars_for_slide:
                    # Use LLM to intelligently shorten section title
                    title_text = await llm_rewrite_for_slide(title_text, max_title_chars_for_slide, text_type="title")

                # Calculate dynamic title height based on estimated lines
                # 32pt bold Calibri: empirical testing shows ~5.5 chars per inch
                # This accounts for letter spacing, bold weight, and character width variance
                # Example: "Real Madrid Partnership Activation Ideas" (44 chars) wraps at ~7" width
                chars_per_inch_at_32pt = 5.5  # Empirical estimate for bold 32pt Calibri
                safe_width_inches = safe_title_width / 914400  # Convert EMU to inches
                title_char_width = max(20, int(safe_width_inches * chars_per_inch_at_32pt))
                estimated_lines = max(1, (len(title_text) + title_char_width - 1) // title_char_width)  # Ceiling division

                # PHASE 9 FIX: More generous height per line for breathing room
                # 32pt font = 0.44" height, with 1.2x line spacing = 0.53" per line
                # Add 0.3" padding for safety (text rendering can be larger than font metric)
                line_height_at_32pt = 0.55  # Height per line including spacing
                dynamic_title_height = (estimated_lines * line_height_at_32pt) + 0.3
                # Clamp to reasonable bounds
                dynamic_title_height = max(0.85, min(dynamic_title_height, 2.0))

                logger.debug(
                    f"Dynamic title height: text='{title_text[:30]}...', "
                    f"chars={len(title_text)}, est_lines={estimated_lines}, height={dynamic_title_height:.2f}in"
                )

                section_title_box = slide.shapes.add_textbox(
                    title_left_pos, title_y,
                    safe_title_width, scaled_inches(dynamic_title_height, "h")  # Dynamic height for wrapped titles
                )
                tf = section_title_box.text_frame
                tf.word_wrap = True
                tf.auto_size = MSO_AUTO_SIZE.NONE
                p = tf.paragraphs[0]
                p.text = title_text
                apply_font_to_paragraph(p, heading_font, is_heading=True)
                p.font.size = Pt(32)
                p.font.bold = True
                apply_color_to_paragraph(p, title_color)

                # Calculate title_box_bottom - this is used for content positioning
                # regardless of whether we add an underline
                # PHASE 11 FIX: Use actual title box bottom + consistent gap
                # The title textbox bottom is title_y + actual box height
                actual_title_box_bottom = title_y + section_title_box.height
                # Add breathing room gap below title (0.15" is ~14 points, good visual spacing)
                underline_gap = Inches(0.15)
                title_box_bottom = actual_title_box_bottom + underline_gap

                # Add accent underline for section title when using template
                if use_template_styling and template_theme:
                    try:
                        accent_color = get_template_accent_color(1)
                        # Position underline dynamically based on title height
                        underline = slide.shapes.add_shape(
                            MSO_SHAPE.RECTANGLE,
                            scaled_inches(0.6, "w"), title_box_bottom,
                            scaled_inches(3.0, "w"), Inches(0.04)
                        )
                        underline.fill.solid()
                        underline.fill.fore_color.rgb = accent_color
                        underline.line.fill.background()
                    except Exception as accent_err:
                        logger.debug(f"Could not add accent underline: {accent_err}")

                if header_style == "none" and not use_template_styling:
                    add_accent_elements(slide)

                # Content - prefer revised_content if it has actual content, otherwise use original
                raw_content = section.revised_content if section.revised_content and section.revised_content.strip() else section.content
                content = sanitize_text(raw_content) or ""
                content = strip_markdown(content)

                if len(content) > 3000:
                    content = await llm_condense_text(content, 3000)

                section.rendered_content = content

                # Auto chart/table detection — charts take priority over images
                # When charts are enabled, try chart detection FIRST regardless of image availability
                # If a chart/table is detected, skip the image for this slide
                rendered_as_chart = False
                rendered_as_table = False

                if auto_charts_enabled:
                    try:
                        from backend.services.pptx_chart_generator import PPTXNativeChartGenerator, create_chart_generator

                        chart_data = PPTXNativeChartGenerator.detect_chartable_data(content)
                        if chart_data and PPTXNativeChartGenerator.validate_chart_data(chart_data):
                            theme_colors = {
                                'primary': theme['primary'],
                                'secondary': theme['secondary'],
                                'accent': theme.get('accent', '#E0E1DD'),
                                'text': theme['text'],
                            }
                            chart_gen = create_chart_generator(theme_colors)
                            suggested_type = PPTXNativeChartGenerator.suggest_chart_type(chart_data)

                            chart_gen.add_chart_to_slide(
                                slide,
                                chart_data,
                                chart_type=suggested_type,
                                left=scaled_inches(0.75, "w"),
                                top=scaled_inches(2.0, "h"),
                                width=scaled_inches(11.0, "w"),
                                height=scaled_inches(4.8, "h"),
                                title=None,
                                show_legend=len(chart_data.series) > 1,
                            )
                            rendered_as_chart = True
                            logger.info(f"Rendered chart on slide {current_slide} (type={suggested_type})")
                            # Preserve content as speaker notes so slide info isn't lost
                            try:
                                notes_slide = slide.notes_slide
                                notes_tf = notes_slide.notes_text_frame
                                notes_tf.text = f"Slide content:\n{content[:800]}"
                            except Exception:
                                pass
                    except Exception as e:
                        logger.warning(f"Chart detection failed: {e}")

                if auto_charts_enabled and not rendered_as_chart:
                    try:
                        from backend.services.pptx_table_generator import PPTXTableGenerator

                        table_data = PPTXTableGenerator.detect_tabular_content(content)
                        # Validate table data: reject if cells are too long (prose, not tabular)
                        if table_data and len(table_data) >= 2 and len(table_data[0]) >= 2:
                            total_cells = sum(len(row) for row in table_data)
                            avg_cell_len = sum(len(cell) for row in table_data for cell in row) / max(1, total_cells)
                            if avg_cell_len > 60:
                                logger.debug(f"Table rejected: avg cell length {avg_cell_len:.0f} > 60 (prose, not tabular)")
                                table_data = None
                        if table_data and len(table_data) >= 2 and len(table_data[0]) >= 2:
                            theme_colors = {
                                'primary': theme['primary'],
                                'secondary': theme['secondary'],
                                'accent': theme.get('accent', '#E0E1DD'),
                                'text': theme['text'],
                            }
                            table_gen = PPTXTableGenerator(theme_colors)

                            table_gen.add_table_to_slide(
                                slide,
                                table_data,
                                left=scaled_inches(0.75, "w"),
                                top=scaled_inches(2.0, "h"),
                                width=scaled_inches(11.0, "w"),
                                has_header=True,
                                font_size=12 if len(table_data) > 8 else 14,
                            )
                            rendered_as_table = True
                            logger.info(f"Rendered table on slide {current_slide}")
                    except Exception as e:
                        logger.warning(f"Table detection failed: {e}")

                # Skip bullet rendering if chart/table rendered
                if rendered_as_chart or rendered_as_table:
                    add_footer(slide, current_slide, total_slides)
                    continue

                # If chart/table was rendered, don't show image (chart takes priority)
                effective_has_image = has_image and not rendered_as_chart and not rendered_as_table

                # Content layout
                slide_content_width = prs.slide_width.inches - 1.6
                content_width_ratio = layout_config.get("content_width", 0.85)

                image_left = Inches(8.5)
                image_width = Inches(4.3)
                # PHASE 10 FIX: Content must start with adequate gap below title/underline
                # title_box_bottom is where underline is placed
                # We need at least 0.35" gap AFTER the underline for proper visual separation
                # This accounts for the underline height (0.04") plus breathing room
                content_top = title_box_bottom + Inches(0.35)  # Increased from 0.15" to 0.35"

                # Cap content height so it never extends into the footer zone
                footer_zone_top = prs.slide_height - Inches(0.45)  # Reserve 0.45" for footer
                max_content_height = footer_zone_top - content_top

                if effective_has_image:
                    image_pos = layout_config.get("image_position", "right")
                    content_height = min(scaled_inches(4.8, "h"), max_content_height)

                    if layout_key == "two_column":
                        content_width = Inches(slide_content_width * 0.48)
                        content_left = scaled_inches(0.8, "w")
                        image_left = scaled_inches(7.2, "w")
                        image_width = scaled_inches(5.5, "w")
                        content_height = min(scaled_inches(5.0, "h"), max_content_height)
                    elif layout_key == "image_focused":
                        content_width = Inches(slide_content_width)
                        content_left = scaled_inches(0.8, "w")
                        content_height = min(scaled_inches(2.0, "h"), max_content_height)
                        image_left = scaled_inches(2.5, "w")
                        image_width = scaled_inches(8.0, "w")
                    elif layout_key == "minimal":
                        content_width = scaled_inches(6.0, "w")
                        content_left = scaled_inches(0.6, "w")
                        image_left = scaled_inches(7.2, "w")
                        image_width = scaled_inches(5.3, "w")
                        content_height = min(scaled_inches(5.0, "h"), max_content_height)
                    else:  # standard
                        content_width = scaled_inches(6.0, "w")
                        content_left = scaled_inches(0.6, "w")
                        image_left = scaled_inches(7.0, "w")
                        image_width = scaled_inches(5.5, "w")
                        content_height = min(scaled_inches(5.0, "h"), max_content_height)

                    content_box = slide.shapes.add_textbox(
                        content_left, content_top,
                        content_width, content_height
                    )
                    # Enable autofit for image-layout content boxes too
                    tf_temp = content_box.text_frame
                    tf_temp.word_wrap = True
                    # PHASE 9 FIX: Use 75% minimum font scale instead of 60%
                    # 60% was too aggressive and made text unreadable
                    # 75% maintains readability at presentation distance
                    enable_text_autofit(tf_temp, min_font_scale=75, max_line_reduction=10)
                else:
                    if layout_key == "minimal":
                        content_width = Inches(slide_content_width * 0.7)
                        content_left = Inches((prs.slide_width.inches - content_width.inches) / 2)
                    elif layout_key == "two_column":
                        content_width = Inches(slide_content_width)
                        content_left = scaled_inches(0.8, "w")
                    else:
                        content_width = scaled_inches(11.5, "w")
                        content_left = scaled_inches(0.8, "w")

                    content_box = slide.shapes.add_textbox(
                        content_left, content_top,  # Use same content_top as image branch
                        content_width, min(scaled_inches(4.8, "h"), max_content_height)
                    )

                tf = content_box.text_frame
                tf.word_wrap = True
                tf.auto_size = MSO_AUTO_SIZE.NONE

                # Set vertical alignment for professional polish
                set_vertical_alignment(tf, 'top')

                # Enable text auto-fit to prevent overflow (instead of truncation)
                # This uses PowerPoint's native font scaling when content is too long
                # PHASE 9 FIX: Use 75% minimum to maintain readability (was 60%)
                enable_text_autofit(tf, min_font_scale=75, max_line_reduction=10)

                # Parse bullet hierarchy
                paragraphs = content.split('\n')

                def parse_bullet_hierarchy(lines: list) -> list:
                    result = []
                    skip_patterns = [
                        r'^[Hh]ere (are|is) ',
                        r'^[Ll]et me ',
                        r'^[Cc]ertainly',
                        r'^[Ss]ure[,!]',
                        r'^[Ii]\'ll ',
                        r'^[Bb]elow (are|is)',
                        r'[Ll]et me know',
                        r'[Hh]ope this helps',
                        r'[Ff]eel free to',
                    ]
                    for line in lines:
                        if not line:
                            continue
                        stripped = line.lstrip()
                        if not stripped:
                            continue
                        if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in skip_patterns):
                            continue
                        indent = len(line) - len(stripped)
                        level = min(indent // 2, 3)
                        if stripped.startswith(('- ', '• ', '* ', '◦ ', '○ ', '▪ ', '▸ ')):
                            text = stripped[2:].strip()
                            if stripped.startswith(('◦ ', '○ ')) and level == 0:
                                level = 1
                        elif stripped.startswith(tuple(f'{i}.' for i in range(1, 10))):
                            text = stripped.split('.', 1)[1].strip() if '.' in stripped else stripped
                        else:
                            text = stripped
                        if text:
                            result.append((text, level))
                    return result

                valid_paragraphs = parse_bullet_hierarchy(paragraphs)

                # CRITICAL FIX: Filter out bullets that duplicate the slide/document title
                # This prevents content like "• Campaign Ideas" appearing as a bullet when
                # "Campaign Ideas" is already the document/slide title
                def filter_title_duplicates(bullets: list, slide_title: str, doc_title: str) -> list:
                    """Remove bullets that are just the section/slide/document title repeated."""
                    if not slide_title and not doc_title:
                        return bullets

                    titles_to_filter = []
                    if slide_title:
                        titles_to_filter.append(slide_title.lower().strip())
                    if doc_title:
                        titles_to_filter.append(doc_title.lower().strip())

                    filtered = []
                    for text, level in bullets:
                        text_lower = text.lower().strip()
                        is_duplicate = False
                        for title in titles_to_filter:
                            # Exact match or bullet starts with short title (min 15 chars)
                            if text_lower == title:
                                is_duplicate = True
                                break
                            if len(title) >= 15 and text_lower.startswith(title[:15]):
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            filtered.append((text, level))

                    if len(filtered) < len(bullets):
                        logger.debug(f"Filtered {len(bullets) - len(filtered)} duplicate title bullets")
                    return filtered

                valid_paragraphs = filter_title_duplicates(
                    valid_paragraphs,
                    slide_title=section.title or "",
                    doc_title=job.title or ""
                )

                # PHASE 9 FIX: Filter out empty/placeholder bullets
                # Remove bullets that are just labels without actual content (e.g., "Idea:", "Ideas", "Key Points:")
                def filter_empty_bullets(bullets: list) -> list:
                    """Remove bullets that are just labels/placeholders without real content."""
                    # Pattern for label-only bullets: ends with colon or is a short generic word
                    label_patterns = [
                        r'^ideas?:?$',           # "Idea:", "Ideas:", "Idea", "Ideas"
                        r'^key\s*points?:?$',    # "Key Points:", "Key Point:"
                        r'^overview:?$',         # "Overview:"
                        r'^summary:?$',          # "Summary:"
                        r'^highlights?:?$',      # "Highlight:", "Highlights:"
                        r'^benefits?:?$',        # "Benefit:", "Benefits:"
                        r'^features?:?$',        # "Feature:", "Features:"
                        r'^objectives?:?$',      # "Objective:", "Objectives:"
                        r'^goals?:?$',           # "Goal:", "Goals:"
                        r'^actions?:?$',         # "Action:", "Actions:"
                        r'^steps?:?$',           # "Step:", "Steps:"
                        r'^notes?:?$',           # "Note:", "Notes:"
                    ]

                    filtered = []
                    for text, level in bullets:
                        text_clean = text.lower().strip()
                        # Skip bullets shorter than 15 chars that match label patterns
                        is_label_only = False
                        if len(text_clean) < 20:
                            for pattern in label_patterns:
                                if re.match(pattern, text_clean, re.IGNORECASE):
                                    is_label_only = True
                                    logger.debug(f"Filtered label-only bullet: '{text}'")
                                    break
                        if not is_label_only and len(text.strip()) > 5:
                            filtered.append((text, level))

                    return filtered

                valid_paragraphs = filter_empty_bullets(valid_paragraphs)

                # Use learned constraints or defaults for max bullets
                if slide_content_planner:
                    max_paras = max_bullets_for_slide
                elif has_image and layout_key == "image_focused":
                    max_paras = 5
                elif has_image:
                    max_paras = 7
                else:
                    max_paras = 12

                total_bullets = min(len(valid_paragraphs), max_paras)
                total_chars = sum(len(text) for text, _ in valid_paragraphs[:max_paras])

                # Character limits adjusted for proper text fitting
                # DYNAMIC CALCULATION: Calculate max_chars based on actual content_width
                # Using ~9 chars per inch at 18pt font as baseline
                # content_width is always an Inches object from python-pptx
                try:
                    content_width_inches = content_width.inches
                except AttributeError:
                    # Fallback if somehow not an Inches object (EMU int value)
                    content_width_inches = content_width / 914400 if isinstance(content_width, (int, float)) else 6.0

                # ENTERPRISE TYPOGRAPHY: Determine font size based on content density
                # Minimum 18pt for readability at presentation distance (consulting standard)
                # McKinsey/BCG use 24pt+ for body, but we allow 18pt minimum for dense slides
                ENTERPRISE_MIN_FONT_SIZE = 18  # Minimum readable at 10+ feet

                if total_bullets <= 4 and total_chars < 300:
                    # Sparse content: larger, more impactful font
                    base_font_size = Pt(24)
                    chars_per_inch = 7
                    line_spacing = Pt(12)
                elif total_bullets <= 6 and total_chars < 500:
                    # Standard 6x6 compliant: comfortable reading
                    base_font_size = Pt(22)
                    chars_per_inch = 8
                    line_spacing = Pt(10)
                elif total_bullets <= 8 and total_chars < 700:
                    # Slightly dense: still professional
                    base_font_size = Pt(20)
                    chars_per_inch = 9
                    line_spacing = Pt(8)
                else:
                    # Dense content: enterprise minimum
                    base_font_size = Pt(ENTERPRISE_MIN_FONT_SIZE)
                    chars_per_inch = 10
                    line_spacing = Pt(6)
                    logger.info(
                        f"Using minimum enterprise font size ({ENTERPRISE_MIN_FONT_SIZE}pt) for dense content",
                        bullets=total_bullets,
                        chars=total_chars,
                    )

                # Calculate max_chars based on actual content box width
                max_chars = int(content_width_inches * chars_per_inch)
                # Apply reasonable bounds
                max_chars = max(40, min(max_chars, 120))

                logger.debug(
                    f"Dynamic max_chars calculation: width={content_width_inches:.1f}in, "
                    f"font={base_font_size.pt}pt, chars_per_inch={chars_per_inch}, max_chars={max_chars}"
                )

                # Override with learned constraints if available (use the smaller value)
                if slide_content_planner and max_bullet_chars_for_slide:
                    max_chars = min(max_chars, max_bullet_chars_for_slide)

                first_para_used = False
                para_count = 0

                # Build list of bullet texts for context awareness
                bullet_texts = [text for text, _ in valid_paragraphs[:max_paras]]
                slide_title_for_context = sanitize_text(section.title) or f"Section {section_idx + 1}"

                for idx, (bullet_text, bullet_level) in enumerate(valid_paragraphs):
                    if para_count >= max_paras:
                        break

                    if first_para_used:
                        p = tf.add_paragraph()
                    else:
                        p = tf.paragraphs[0]
                        first_para_used = True
                    para_count += 1

                    if len(bullet_text) > max_chars:
                        # Build context for surrounding bullet awareness
                        bullet_context = {
                            'slide_title': slide_title_for_context,
                            'prev_bullet': bullet_texts[idx - 1] if idx > 0 else None,
                            'next_bullet': bullet_texts[idx + 1] if idx < len(bullet_texts) - 1 else None,
                        }
                        bullet_text = await llm_condense_text(bullet_text, max_chars, context=bullet_context)

                    # Final validation: ensure bullet is a complete thought
                    if not is_sentence_complete(bullet_text):
                        # Try to fix by removing incomplete ending
                        fixed_bullet = ensure_complete_thought(bullet_text, len(bullet_text))
                        if fixed_bullet and is_sentence_complete(fixed_bullet):
                            bullet_text = fixed_bullet
                        else:
                            # Last resort: trim incomplete ending words
                            words = bullet_text.split()
                            while len(words) > 3 and not is_sentence_complete(' '.join(words)):
                                words.pop()
                            bullet_text = ' '.join(words)
                            logger.debug(f"Fixed incomplete bullet: {bullet_text[:50]}...")

                    # Set paragraph level for bullet hierarchy
                    p.level = bullet_level
                    theme_bullets = content_bullets if content_bullets else ['•', '◦', '▪']
                    bullet_char = theme_bullets[min(bullet_level, len(theme_bullets) - 1)]

                    # Use native PowerPoint bullet formatting for proper theme integration
                    # This uses OOXML a:buChar elements instead of manual text prefixes
                    # Benefits: proper hierarchy, theme colors, accessibility, better editing
                    use_native_bullets = job.options.get('use_native_bullets', True) if hasattr(job, 'options') and job.options else True

                    if use_native_bullets:
                        # Native bullet formatting (enterprise best practice)
                        add_native_bullet_to_paragraph(p, bullet_text, level=bullet_level, bullet_char=bullet_char)

                        # Apply left margin for sub-bullets
                        if bullet_level > 0:
                            try:
                                p.paragraph_format.left_margin = Inches(bullet_level * 0.4)
                            except Exception as margin_err:
                                logger.debug(f"Could not set left margin for level {bullet_level}: {margin_err}")
                    else:
                        # Fallback: Manual text prefix (legacy behavior)
                        if bullet_level > 0:
                            try:
                                p.paragraph_format.left_margin = Inches(bullet_level * 0.4)
                            except Exception as margin_err:
                                logger.debug(f"Could not set left margin for level {bullet_level}: {margin_err}")

                        indent_prefix = "  " * bullet_level
                        p.text = f"{indent_prefix}{bullet_char} {bullet_text}"
                    apply_font_to_paragraph(p, body_font)
                    # Increase font size difference between levels for better visual distinction
                    # Use 4pt reduction per level (was 2pt) for clearer hierarchy
                    level_font_size = Pt(max(base_font_size.pt - (bullet_level * 4), 12))
                    p.font.size = level_font_size

                    if use_template_styling:
                        content_text_color = TEXT_COLOR
                    else:
                        content_text_color = get_text_color_for_background(is_content_slide=True)
                    apply_color_to_paragraph(p, content_text_color)

                    # Set line spacing WITHIN wrapped bullets (115% for better readability)
                    p.line_spacing = 1.15
                    # Set spacing BETWEEN bullets
                    p.space_after = line_spacing

                if not first_para_used:
                    p = tf.paragraphs[0]
                    p.text = ""
                    apply_font_to_paragraph(p, body_font)
                    p.font.size = Pt(16)

                # SAFETY NET: Apply fit_text to shrink font if content still overflows
                # This handles edge cases where LLM rewriting and enforcement weren't sufficient
                if para_count > 0:
                    safe_fit_text(
                        tf,
                        font_family=body_font or "Calibri",
                        max_size=int(base_font_size.pt),
                        min_size=12,
                        bold=False,
                    )

                if enable_animations and para_count > 0:
                    add_bullet_animations(slide, content_box, animation_type="appear", delay_between=400)

                # Add image if available
                if has_image and section_idx in section_images:
                    try:
                        image_path = section_images.get(section_idx)
                        image_added = False

                        # Try to use picture placeholder first (for template styling)
                        if use_template_styling:
                            pic_placeholder = find_picture_placeholder(slide)
                            if pic_placeholder:
                                pic = insert_image_into_placeholder(slide, image_path, pic_placeholder)
                                if pic:
                                    image_added = True
                                    logger.debug(f"Used picture placeholder for section {section_idx}")

                        # Fall back to manual positioning if placeholder not used
                        if not image_added:
                            # ACCESSIBILITY: Generate descriptive alt text for the image
                            section_title = section.title[:100] if section.title else f"Section {section_idx + 1}"
                            alt_text = f"Visual illustration for: {section_title}"

                            if layout_key == "image_focused":
                                # Center horizontally, position below content
                                img_w = image_width.inches if hasattr(image_width, 'inches') else 8.0
                                img_h = 3.0
                                centered_x, _ = get_centered_image_position(img_w, img_h)
                                add_picture_with_alt_text(
                                    slide, image_path,
                                    centered_x, scaled_inches(4.2, "h"),
                                    width=image_width,
                                    height=scaled_inches(3.0, "h"),
                                    alt_text=alt_text,
                                )
                            elif layout_key == "two_column":
                                # Two column keeps image on right side
                                add_picture_with_alt_text(
                                    slide, image_path,
                                    image_left, scaled_inches(2.0, "h"),
                                    width=image_width,
                                    height=scaled_inches(4.0, "h"),
                                    alt_text=alt_text,
                                )
                            else:
                                # Standard layout: place image on RIGHT side (not centered)
                                # to avoid overlapping with content text on left
                                # Calculate safe image height that won't overlap footer
                                footer_zone_top = prs.slide_height - Inches(0.45)  # Footer at -0.35, plus 0.1 margin
                                content_top_inches = content_top.inches if hasattr(content_top, 'inches') else 1.5
                                max_safe_height = footer_zone_top - Inches(content_top_inches)
                                image_height = min(scaled_inches(4.5, "h"), max_safe_height)

                                add_picture_with_alt_text(
                                    slide, image_path,
                                    image_left, content_top,
                                    width=image_width,
                                    height=image_height,
                                    alt_text=alt_text,
                                )
                    except Exception as e:
                        logger.warning(f"Failed to add image: {e}")
                        # Clean up any remaining picture placeholder if image failed
                        if use_template_styling:
                            pic_placeholder = find_picture_placeholder(slide)
                            if pic_placeholder:
                                try:
                                    sp = pic_placeholder._element
                                    sp.getparent().remove(sp)
                                except Exception:
                                    pass

                add_footer(slide, current_slide, total_slides)

                # Speaker notes - PHASE 10: Include section explanations when enabled
                include_notes_explanation = job.metadata.get("include_notes_explanation", True)  # Default True
                notes_parts = [f"SECTION: {section.title or f'Section {section_idx + 1}'}", ""]

                # Add section description/explanation if enabled
                if include_notes_explanation:
                    section_desc = getattr(section, 'description', '') or getattr(section, 'revised_content', '') or ''
                    # Use the original content summary if no description available
                    if not section_desc and hasattr(section, 'content') and section.content:
                        # PHASE 12: Increased from 500 to 1500 chars for better content coverage
                        section_desc = section.content[:1500].strip()
                        if len(section.content) > 1500:
                            section_desc += "..."

                    if section_desc:
                        notes_parts.append("SLIDE EXPLANATION:")
                        # PHASE 12: Increased from 800 to 2000 chars to prevent truncation
                        notes_parts.append(f"  {section_desc[:2000]}")
                        notes_parts.append("")

                section_sources = section.sources or []
                if not section_sources and job.sources_used:
                    section_sources = job.sources_used[:5]

                if section_sources:
                    # Group sources by usage type
                    from ...models import SourceUsageType
                    content_sources = [s for s in section_sources if getattr(s, 'usage_type', SourceUsageType.CONTENT) == SourceUsageType.CONTENT]
                    style_sources = [s for s in section_sources if getattr(s, 'usage_type', None) == SourceUsageType.STYLE]
                    other_sources = [s for s in section_sources if getattr(s, 'usage_type', None) not in [SourceUsageType.CONTENT, SourceUsageType.STYLE, None]]

                    def format_source(s, prefix=""):
                        """Format a source reference with page/slide number."""
                        source_name = s.document_name or s.document_id
                        location = ""
                        if s.page_number:
                            if source_name.lower().endswith('.pptx'):
                                location = f" (Slide {s.page_number})"
                            else:
                                location = f" (Page {s.page_number})"
                        usage_desc = ""
                        if hasattr(s, 'usage_description') and s.usage_description:
                            usage_desc = f" - {s.usage_description}"
                        return f"{prefix}• {source_name}{location}{usage_desc}"

                    notes_parts.append("REFERENCES:")

                    if content_sources:
                        notes_parts.append("  Content Sources:")
                        for s in content_sources[:3]:
                            notes_parts.append(format_source(s, "    "))

                    if style_sources:
                        notes_parts.append("  Style References:")
                        for s in style_sources[:2]:
                            notes_parts.append(format_source(s, "    "))

                    if other_sources:
                        notes_parts.append("  Other References:")
                        for s in other_sources[:2]:
                            usage_type = getattr(s, 'usage_type', 'reference')
                            notes_parts.append(f"    • {s.document_name or s.document_id} [{usage_type.value if hasattr(usage_type, 'value') else usage_type}]")

                    # If no categorized sources, fall back to simple list
                    if not content_sources and not style_sources and not other_sources:
                        for s in section_sources[:5]:
                            notes_parts.append(format_source(s, "  "))
                else:
                    # No RAG sources found - check if we have a template reference
                    template_filename = job.metadata.get("template_pptx_filename")
                    if template_filename:
                        notes_parts.append("REFERENCES:")
                        notes_parts.append(f"  Style Template: {template_filename}")
                    else:
                        notes_parts.append("REFERENCES: AI-generated content (no source documents indexed)")

                section_notes = "\n".join(notes_parts)
                section_notes += "\n\n---"
                if job.metadata.get("user_email"):
                    section_notes += f"\nUser: {job.metadata.get('user_email')}"
                section_notes += f"\nModel: {job.metadata.get('llm_model', 'N/A')}"
                section_notes += f"\nSlide: {current_slide} / {total_slides}"

                add_slide_notes(slide, section_notes)

            # ========== SOURCES SLIDE ==========
            logger.info(
                "Sources slide check",
                include_sources=include_sources,
                sources_count=len(job.sources_used) if job.sources_used else 0,
                sources_preview=[
                    {
                        "name": s.document_name,
                        "page": s.page_number,
                        "doc_id": s.document_id[:20] if s.document_id else None
                    }
                    for s in (job.sources_used or [])[:3]
                ],
            )
            if include_sources and job.sources_used:
                current_slide += 1
                sources_layout = get_slide_layout("sources")
                slide = prs.slides.add_slide(sources_layout)
                remove_empty_placeholders(slide)

                # Copy branding elements (logos, backgrounds) from layout to slide
                if use_template_styling:
                    copy_layout_branding_to_slide(slide, sources_layout)

                if enable_animations:
                    add_slide_transition(slide, "blinds", duration=transition_duration, speed=animation_speed)

                if not use_template_styling:
                    add_header_accent(slide, "Sources & References")

                title_y = scaled_inches(0.3, "h") if header_style != "none" else scaled_inches(0.5, "h")
                if use_template_styling:
                    title_color = TEXT_COLOR
                else:
                    title_color = WHITE if header_style != "none" else TEXT_COLOR

                # Calculate safe title width respecting logo zone (same logic as content slides)
                sources_title_width = prs.slide_width - Inches(3.0)  # Default: leave 3" right margin for logo
                if slide_content_planner and hasattr(slide_content_planner.learner, 'detected_logo_x'):
                    logo_x = slide_content_planner.learner.detected_logo_x
                    if logo_x is not None:
                        sources_title_width = logo_x - scaled_inches(0.8, "w") - Inches(0.3)
                sources_title_width = max(Inches(4.0), min(sources_title_width, prs.slide_width - Inches(3.0)))

                sources_title = slide.shapes.add_textbox(
                    scaled_inches(0.8, "w"), title_y,
                    sources_title_width, scaled_inches(0.8, "h")
                )
                tf = sources_title.text_frame
                tf.word_wrap = True
                tf.auto_size = MSO_AUTO_SIZE.NONE
                p = tf.paragraphs[0]
                p.text = "Sources & References"
                apply_font_to_paragraph(p, heading_font, is_heading=True)
                p.font.size = Pt(32)
                p.font.bold = True
                apply_color_to_paragraph(p, title_color)

                if header_style == "none" and not use_template_styling:
                    add_accent_elements(slide)

                sources_box = slide.shapes.add_textbox(
                    scaled_inches(0.8, "w"), scaled_inches(2.0, "h"),
                    prs.slide_width - Inches(1.6), scaled_inches(4.8, "h")
                )
                tf = sources_box.text_frame
                tf.word_wrap = True
                tf.auto_size = MSO_AUTO_SIZE.NONE

                if use_template_styling:
                    sources_text_color = TEXT_COLOR
                else:
                    sources_text_color = get_text_color_for_background(is_content_slide=True)

                source_bullet = content_bullets[0] if content_bullets else "•"

                # Group sources by usage type for the sources slide
                from ...models import SourceUsageType
                content_sources = [s for s in job.sources_used if getattr(s, 'usage_type', SourceUsageType.CONTENT) == SourceUsageType.CONTENT]
                style_sources = [s for s in job.sources_used if getattr(s, 'usage_type', None) == SourceUsageType.STYLE]
                other_sources = [s for s in job.sources_used if getattr(s, 'usage_type', None) not in [SourceUsageType.CONTENT, SourceUsageType.STYLE, None]]

                def format_source_text(source):
                    """Format source with location and usage info."""
                    doc_id = source.document_id[:20] if source.document_id else "Unknown"
                    doc_name = sanitize_text(source.document_name or doc_id)
                    location_info = ""
                    if source.page_number:
                        if doc_name.lower().endswith('.pptx'):
                            location_info = f" (Slide {source.page_number})"
                        else:
                            location_info = f" (Page {source.page_number})"
                    return f"{source_bullet}  {doc_name}{location_info}"

                def add_source_paragraph(source, is_first):
                    """Add a paragraph for a source with optional hyperlink."""
                    nonlocal first_source_used
                    if is_first and not first_source_used:
                        p = tf.paragraphs[0]
                        first_source_used = True
                    else:
                        p = tf.add_paragraph()

                    # Check for hyperlink URL (document_url or file:// path)
                    hyperlink_url = None
                    if hasattr(source, 'document_url') and source.document_url:
                        hyperlink_url = source.document_url
                    elif hasattr(source, 'document_path') and source.document_path:
                        # Convert local path to file:// URL
                        import urllib.parse
                        hyperlink_url = f"file://{urllib.parse.quote(source.document_path)}"

                    if hyperlink_url:
                        # Create run with hyperlink
                        p.clear()
                        run = p.add_run()
                        run.text = format_source_text(source)
                        run.font.name = body_font
                        run.font.size = Pt(14)
                        run.font.color.rgb = sources_text_color
                        try:
                            run.hyperlink.address = hyperlink_url
                            # Style as hyperlink (underline)
                            run.font.underline = True
                        except Exception as link_err:
                            logger.debug(f"Could not add source hyperlink: {link_err}")
                    else:
                        p.text = format_source_text(source)
                        apply_font_to_paragraph(p, body_font)
                        p.font.size = Pt(14)
                        apply_color_to_paragraph(p, sources_text_color)

                    p.space_after = Pt(6)
                    return p

                def add_section_header(text, is_first):
                    """Add a section header for source categories."""
                    nonlocal first_source_used
                    if is_first and not first_source_used:
                        p = tf.paragraphs[0]
                        first_source_used = True
                    else:
                        p = tf.add_paragraph()
                    p.text = text
                    apply_font_to_paragraph(p, body_font)
                    p.font.size = Pt(12)
                    p.font.bold = True
                    apply_color_to_paragraph(p, sources_text_color)
                    p.space_before = Pt(8)
                    p.space_after = Pt(4)
                    return p

                first_source_used = False

                # Add content sources
                if content_sources:
                    add_section_header("Content References:", not first_source_used)
                    for source in content_sources[:6]:
                        add_source_paragraph(source, False)

                # Add style sources
                if style_sources:
                    add_section_header("Style References:", not first_source_used)
                    for source in style_sources[:3]:
                        add_source_paragraph(source, False)

                # Add other sources
                if other_sources:
                    add_section_header("Other References:", not first_source_used)
                    for source in other_sources[:3]:
                        add_source_paragraph(source, False)

                # Fallback if no categorized sources
                if not content_sources and not style_sources and not other_sources:
                    for source in job.sources_used[:10]:
                        add_source_paragraph(source, not first_source_used)

                if not first_source_used:
                    p = tf.paragraphs[0]
                    p.text = "No sources available"
                    apply_font_to_paragraph(p, body_font)
                    p.font.size = Pt(14)

                if enable_animations and first_source_used:
                    add_bullet_animations(slide, sources_box, animation_type="appear", delay_between=250)

                add_footer(slide, current_slide, total_slides)

            # ========== POST-PROCESSING: ADD TOC HYPERLINKS & PAGE NUMBERS ==========
            if config.include_toc and toc_paragraphs and content_slides_by_section:
                try:
                    # Calculate slide numbers for each section
                    # Title slide = 1, TOC = 2, Exec Summary = 3, then content slides
                    base_slide_offset = 1  # 1-indexed
                    if config.include_toc:
                        base_slide_offset += 1
                    if config.include_executive_summary:
                        base_slide_offset += 1
                    base_slide_offset += 1  # Title slide

                    for toc_idx, (toc_para, section_idx) in enumerate(zip(toc_paragraphs, toc_section_indices)):
                        if section_idx in content_slides_by_section:
                            target_slide = content_slides_by_section[section_idx]
                            toc_text = toc_para.text

                            # Calculate the slide number for this section
                            slide_number = base_slide_offset + section_idx

                            toc_para.clear()

                            # Number run with accent color
                            number_run = toc_para.add_run()
                            number_run.text = f"{section_idx + 1}."
                            number_run.font.size = Pt(20)
                            number_run.font.bold = True
                            if use_template_styling:
                                apply_accent_to_run(number_run, accent_num=1)
                            else:
                                number_run.font.color.rgb = ACCENT_COLOR

                            # Title run with hyperlink
                            title_text = toc_text.lstrip('0123456789. ')  # Remove existing number prefix
                            run = toc_para.add_run()
                            run.text = f"  {title_text}"
                            run.font.name = body_font
                            run.font.size = Pt(20)
                            run.font.bold = True
                            run.font.color.rgb = content_text_color
                            run.font.underline = False

                            try:
                                rId = run.part.relate_to(target_slide.part, RT.SLIDE)
                                hlink = run.hyperlink._hlinkClick
                                hlink.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id', rId)
                                hlink.set('action', 'ppaction://hlinksldjump')
                            except Exception as link_err:
                                logger.debug(f"Could not add hyperlink: {link_err}")

                            # Add page number indicator
                            page_run = toc_para.add_run()
                            page_run.text = f"  —  {slide_number}"
                            page_run.font.size = Pt(14)
                            page_run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
                            if body_font:
                                page_run.font.name = body_font

                    logger.info("Added TOC hyperlinks and page numbers", count=len(toc_paragraphs))
                except Exception as e:
                    logger.warning(f"Failed to add TOC hyperlinks: {e}")

            # Save presentation
            # Note: filename already includes extension from service._generate_filename()
            output_path = os.path.join(config.output_dir, filename)

            # Set document properties
            try:
                prs.core_properties.title = job.title or filename
                prs.core_properties.author = job.metadata.get("user_email", "Document Generator")

                template_filename = job.metadata.get("template_pptx_filename")
                if template_filename:
                    prs.core_properties.comments = f"Template: {template_filename}"
            except Exception as prop_err:
                logger.warning(f"Could not set document properties: {prop_err}")

            prs.save(output_path)

            logger.info("PPTX generated", path=output_path)

            # ========== VISION-BASED SLIDE REVIEW ==========
            # PHASE 10: Enabled by default to catch positioning and quality issues
            enable_vision_review = job.metadata.get(
                "enable_vision_review",
                getattr(config, 'enable_vision_review', True)  # Changed to True by default
            )
            if enable_vision_review:
                try:
                    vision_model = job.metadata.get(
                        "vision_review_model",
                        getattr(config, 'vision_review_model', 'auto')
                    )
                    review_all_slides = job.metadata.get(
                        "vision_review_all_slides",
                        getattr(config, 'vision_review_all_slides', False)
                    )
                    await _run_vision_review(
                        output_path=output_path,
                        job=job,
                        sections=job.sections,
                        vision_model=vision_model,
                        review_all_slides=review_all_slides,
                    )
                except Exception as vision_err:
                    # Vision review is optional - don't fail the entire generation
                    logger.warning(f"Vision review failed (non-fatal): {vision_err}")

            return output_path

        except ImportError as e:
            logger.error(f"python-pptx import error: {e}")
            raise
