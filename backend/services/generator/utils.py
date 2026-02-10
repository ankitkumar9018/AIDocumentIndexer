"""
Document Generation Utilities

Text processing and utility functions for document generation.
Extracted from generator.py for modularity.
"""

import re
from typing import Optional, List, Any

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Text Processing Utilities
# =============================================================================

def strip_markdown(text: str) -> str:
    """Remove markdown formatting, returning clean text.

    Used by PPTX, XLSX, and other generators that don't support markdown.
    """
    if not text:
        return ""
    # Remove headers at start of line (# ## ### #### etc.)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Also remove ### anywhere in content (LLM sometimes outputs "### Title" in bullets)
    text = re.sub(r'\s*#{1,6}\s+', ' ', text)
    # Remove bold **text** or __text__
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    # Remove italic *text* or _text_ (but not bullet points)
    text = re.sub(r'(?<!\s)\*([^*\n]+)\*(?!\s)', r'\1', text)
    text = re.sub(r'(?<!\s)_([^_\n]+)_(?!\s)', r'\1', text)
    # Remove code backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove link syntax [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Clean up multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def filter_llm_metatext(text: str) -> str:
    """Remove common LLM conversational artifacts from generated content.

    Filters out:
    - Preamble text like "Here are the bullet points for..."
    - Closing remarks like "Let me know if you need any adjustments"
    - Conversational artifacts like "Certainly, here's...", "Sure, here are..."
    """
    if not text:
        return ""

    # Patterns to remove
    patterns = [
        # Preamble patterns (at start of content)
        r'^.*?[Hh]ere (?:are|is) (?:the )?.*?:\s*\n?',           # "Here are the bullet points:"
        r'^.*?[Ll]et me (?:provide|create|write|explain).*?:\s*\n?',  # "Let me provide..."
        r'^.*?[Cc]ertainly[,!]?\s*[Hh]ere.*?:\s*\n?',              # "Certainly, here's..."
        r'^.*?[Ss]ure[,!]?\s*[Hh]ere.*?:\s*\n?',                   # "Sure, here are..."
        r'^.*?[Ii]\'ll (?:provide|create|write|give).*?:\s*\n?',  # "I'll provide..."
        r'^.*?[Bb]elow (?:are|is).*?:\s*\n?',                      # "Below are the..."
        # Closing patterns (at end of content)
        r'\n?[Ll]et me know if you (?:need|want|have).*$',        # "Let me know if you need..."
        r'\n?[Hh]ope this helps.*$',                              # "Hope this helps!"
        r'\n?[Ff]eel free to (?:ask|reach|contact).*$',           # "Feel free to ask..."
        r'\n?[Ii]f you (?:have|need) any (?:questions|changes).*$',  # "If you have any questions..."
        r'\n?[Pp]lease let me know.*$',                           # "Please let me know..."
        # Section metadata echoed as content
        r'^\s*[-•*▪◦▸]?\s*\(Section \d+ of \d+\)\s*$',           # "(Section 1 of 8)" standalone
        # Preamble with "we will" / "to measure"
        r'^.*?[Ww]e (?:will|\'ll) (?:track|provide|create|use|measure).*?:\s*\n?',  # "We will track..."
        r'^.*?[Tt]o (?:measure|track|monitor) (?:the )?success.*?:\s*\n?',  # "To measure the success..."
        # Style instructions appearing as content (LLM outputting the style guide)
        r'^\s*[-•*▪◦▸]?\s*(?:Writing )?[Ss]tyle [Rr]equirements?.*$',  # "Writing Style Requirements"
        r'^\s*[-•*▪◦▸]?\s*[Mm]aintain a professional tone.*$',    # "Maintain a professional tone..."
        r'^\s*[-•*▪◦▸]?\s*[Uu]se (?:simple|clear|concise) language.*$',  # "Use simple language..."
        r'^\s*[-•*▪◦▸]?\s*[Tt]he (?:new )?content should (?:use|have|be|match).*$',  # "The content should use..."
        r'^\s*[-•*▪◦▸]?\s*[Kk]ey characteristics of the desired writing style.*$',  # "Key characteristics..."
        r'^\s*[-•*▪◦▸]?\s*[Uu]sing medium-length sentences.*$',   # "Using medium-length sentences..."
        r'^\s*[-•*▪◦▸]?\s*[Ii]ncorporating action verbs.*$',      # "Incorporating action verbs..."
        r'^\s*[-•*▪◦▸]?\s*[Bb]y following these style requirements.*$',  # "By following these style requirements..."
        # Preamble "To [action]..." patterns that describe what will be done
        r'^\s*[-•*▪◦▸]?\s*[Tt]o (?:create|establish|build|develop|implement|ensure|achieve) (?:a |the )?(?:buzz|strong|solid|effective|successful).*?,\s*we\s+.*$',  # "To create buzz..., we..."
        r'^\s*[-•*▪◦▸]?\s*[Tt]o (?:create|establish|build|develop|implement|ensure|achieve).*?:\s*$',  # "To create buzz:" followed by colon
        # In order to patterns
        r'^\s*[-•*▪◦▸]?\s*[Ii]n order to (?:create|establish|build|develop|implement|ensure|achieve).*?:\s*$',  # "In order to..."
        # For the purpose patterns
        r'^\s*[-•*▪◦▸]?\s*[Ff]or the purpose of (?:creating|establishing|building|developing).*?:\s*$',  # "For the purpose of..."
    ]

    result = text
    for pattern in patterns:
        result = re.sub(pattern, '', result, flags=re.MULTILINE | re.IGNORECASE)

    return result.strip()


def filter_title_echo(content: str, section_title: str) -> str:
    """Remove bullet points that just echo the section title.

    Filters out lines where the bullet text matches the section title,
    including variants with section count suffix like "(Section 1 of 8)".
    """
    if not content or not section_title:
        return content

    # Normalize section title for comparison
    title_normalized = section_title.upper().strip()
    # Also handle roman numeral prefixes (I., II., III., etc.)
    title_no_roman = re.sub(r'^[IVXLCDM]+\.\s*', '', title_normalized)

    lines = content.split('\n')
    filtered = []

    for line in lines:
        # Strip bullet markers for comparison
        text = re.sub(r'^[-•*▪◦▸]\s*', '', line.strip())
        # Remove section count suffix like "(Section 1 of 8)"
        text = re.sub(r'\s*\(Section \d+ of \d+\)\s*$', '', text, flags=re.IGNORECASE)
        text_normalized = text.upper().strip()

        # Also remove roman numeral prefix from text
        text_no_roman = re.sub(r'^[IVXLCDM]+\.\s*', '', text_normalized)

        # Skip if it matches the title (with or without roman numerals)
        if text_normalized == title_normalized or text_no_roman == title_no_roman:
            continue
        if text_normalized == title_no_roman or text_no_roman == title_normalized:
            continue

        # Keep the line
        filtered.append(line)

    return '\n'.join(filtered)


def validate_language_purity(content: str, target_language: str = "en") -> tuple[str, List[str]]:
    """Validate that content is in the target language and filter out foreign words.

    This helps address issue of LLM code-switching (mixing languages) when source
    documents contain multiple languages.

    Args:
        content: The generated content to validate
        target_language: The target language code (default 'en' for English)

    Returns:
        Tuple of (cleaned_content, list_of_removed_foreign_words)
    """
    if not content:
        return content, []

    removed_words = []

    # Common foreign word patterns that shouldn't appear in English output
    # These are words that commonly leak from German/other language sources
    foreign_word_patterns = []

    if target_language == "en":
        foreign_word_patterns = [
            # German words that commonly leak into English output
            r'\bAUFGABENSTELLUNG\b',  # task assignment
            r'\bENTWICKLUNG\b',  # development
            r'\bEINER\b',  # a/an
            r'\bDIE\b(?!\s+(Hard|Way|Out))',  # the (but not "Die Hard", etc.)
            r'\bDAS\b',  # the/it
            r'\bUND\b',  # and
            r'\bMIT\b(?!\s)',  # with
            r'\bFÜR\b',  # for
            r'\bZUM\b',  # to the
            r'\bZUR\b',  # to the
            r'\bKONZEPT\b',  # concept
            r'\bSTRATEGIE\b(?!s?\b)',  # strategy (allow English "strategies")
            r'\bMARKE\b',  # brand
            r'\bKAMPAGNE\b',  # campaign
            r'\bBEREICH\b',  # area/domain
            r'\bERGEBNIS\b',  # result
            r'\bANFORDERUNG\b',  # requirement
            # French words that commonly leak into English output
            r'\bLES\b(?!\s+(Paul|Mis))',  # the (but not "Les Paul", "Les Mis")
            r'\bDES\b(?!\s)',  # of the
            r'\bPOUR\b',  # for
            r'\bDANS\b',  # in
            r'\bAVEC\b',  # with
            r'\bENTREPRISE\b',  # enterprise/company
            r'\bDÉVELOPPEMENT\b',  # development
            r'\bSTRATÉGIE\b',  # strategy
            r'\bOBJECTIF\b(?!e)',  # objective (not English "objective")
            r'\bRÉSULTAT\b',  # result
            # Spanish words that commonly leak into English output
            r'\bLOS\b(?!\s+Angeles)',  # the (but not "Los Angeles")
            r'\bDEL\b(?!\s)',  # of the
            r'\bPARA\b',  # for
            r'\bCON\b(?!\s)',  # with
            r'\bEMPRESA\b',  # company
            r'\bDESARROLLO\b',  # development
            r'\bESTRATEGIA\b',  # strategy
            r'\bOBJETIVO\b',  # objective
            r'\bRESULTADO\b',  # result
        ]

    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        cleaned_line = line
        for pattern in foreign_word_patterns:
            matches = re.findall(pattern, cleaned_line, re.IGNORECASE)
            if matches:
                removed_words.extend(matches)
                # Remove the foreign word (may leave awkward spacing, but better than mixed language)
                cleaned_line = re.sub(pattern, '', cleaned_line, flags=re.IGNORECASE)

        # Clean up resulting double spaces or orphaned punctuation
        cleaned_line = re.sub(r'\s{2,}', ' ', cleaned_line)
        cleaned_line = re.sub(r'^\s*[,;:]\s*', '', cleaned_line)

        cleaned_lines.append(cleaned_line)

    cleaned_content = '\n'.join(cleaned_lines)

    if removed_words:
        logger.info(
            "Removed foreign language words from content",
            target_language=target_language,
            removed_words=removed_words[:10],  # Log first 10
            total_removed=len(removed_words)
        )

    return cleaned_content, removed_words


def is_sentence_complete(text: str) -> bool:
    """Check if a text snippet is a complete sentence.

    Helps detect truncated or incomplete bullets that were cut off mid-sentence.

    Args:
        text: The text to check

    Returns:
        True if the text appears to be a complete sentence
    """
    if not text or len(text.strip()) < 5:
        return False

    text = text.strip()

    # Check for common incomplete sentence patterns
    incomplete_patterns = [
        r'\s+for\s+a\s*$',  # "... for a"
        r'\s+to\s+a\s*$',  # "... to a"
        r'\s+with\s+a\s*$',  # "... with a"
        r'\s+the\s*$',  # "... the"
        r'\s+and\s+its?\s*$',  # "... and it/its"
        r'\s+to\s*$',  # "... to"
        r'\s+for\s*$',  # "... for"
        r'\s+with\s*$',  # "... with"
        r'\s+in\s*$',  # "... in"
        r'\s+on\s*$',  # "... on"
        r'\s+by\s*$',  # "... by"
        r'\s+a\s*$',  # "... a"
        r'\s+an\s*$',  # "... an"
        r'\s+that\s*$',  # "... that"
        r'\s+which\s*$',  # "... which"
        r'\s+create\s*$',  # "... create"
        r'\s+develop\s*$',  # "... develop"
        r'\s+unique\s*$',  # "... unique" (likely incomplete)
    ]

    for pattern in incomplete_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    # Check if ends with proper punctuation or is a reasonable fragment
    # Bullets don't always need periods, but should end on a complete word
    last_word = text.split()[-1] if text.split() else ''

    # Common prepositions/articles that shouldn't end a sentence
    bad_endings = {'a', 'an', 'the', 'to', 'for', 'with', 'in', 'on', 'by', 'and', 'or', 'its', 'it', 'their'}

    if last_word.lower().rstrip('.,;:!?') in bad_endings:
        return False

    return True


def filter_incomplete_sentences(content: str) -> str:
    """Filter out bullet points with incomplete sentences.

    Identifies and removes bullets that appear to be cut off mid-sentence.

    Args:
        content: Content with bullet points

    Returns:
        Content with incomplete bullets removed or flagged
    """
    if not content:
        return content

    lines = content.split('\n')
    filtered_lines = []
    removed_count = 0

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            filtered_lines.append(line)
            continue

        # Check if it's a bullet point
        is_bullet = stripped.startswith(('-', '•', '*', '◦', '▪', '▸'))

        if is_bullet:
            # Extract bullet text (remove bullet marker)
            bullet_text = re.sub(r'^[-•*◦▪▸]\s*', '', stripped)

            if not is_sentence_complete(bullet_text):
                logger.debug(
                    "Filtered incomplete bullet",
                    bullet_text=bullet_text[:50]
                )
                removed_count += 1
                continue  # Skip this incomplete bullet

        filtered_lines.append(line)

    if removed_count > 0:
        logger.info(
            "Removed incomplete sentences from content",
            removed_count=removed_count
        )

    return '\n'.join(filtered_lines)


def smart_truncate(text: str, max_chars: int) -> str:
    """Truncate text at word boundaries to avoid mid-word cuts."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.7:  # Don't cut too much
        return truncated[:last_space] + '...'
    return truncated + '...'


def sentence_truncate(text: str, max_chars: int) -> str:
    """Truncate text at sentence boundaries, avoiding mid-sentence cuts.

    Handles common abbreviations and ensures meaningful content is preserved.
    Falls back to clause boundaries, then word boundaries.

    Args:
        text: Text to truncate
        max_chars: Maximum character limit

    Returns:
        Truncated text ending at a complete sentence, clause, or word boundary
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    # Common abbreviations that shouldn't be treated as sentence ends
    # Use negative lookbehind to exclude these
    abbreviations = r'(?<![Mm]r)(?<![Mm]rs)(?<![Mm]s)(?<![Dd]r)(?<![Pp]rof)(?<![Ii]nc)(?<![Ll]td)(?<![Jj]r)(?<![Ss]r)(?<![Vv]s)(?<!etc)(?<!e\.g)(?<!i\.e)'

    # Find sentence endings: period/exclamation/question followed by space or end
    # Exclude common abbreviations
    pattern = abbreviations + r'[.!?](?:\s|$)'
    sentence_ends = list(re.finditer(pattern, truncated))

    if sentence_ends:
        last_end = sentence_ends[-1].end()
        # Use 50% threshold to preserve more complete sentences
        if last_end > max_chars * 0.5:
            return text[:last_end].strip()

    # Fallback: find last complete clause (before comma, semicolon, colon, or dash)
    clause_ends = list(re.finditer(r'[,;:—–-]\s', truncated))
    if clause_ends and clause_ends[-1].start() > max_chars * 0.5:
        return text[:clause_ends[-1].start()].strip() + '...'

    # Final fallback to word boundary
    return smart_truncate(text, max_chars)


async def llm_condense_text(
    text: str,
    max_chars: int,
    fallback_truncate: bool = True,
    preserve_numbers: bool = True,
    context_type: str = "bullet_point",
) -> str:
    """Use LLM to condense text while preserving meaning and critical data.

    Instead of truncating with '...', this function uses LLM to intelligently
    rephrase the text to fit within the character limit while preserving
    numbers, percentages, statistics, and key facts.

    Args:
        text: The text to condense
        max_chars: Maximum allowed characters
        fallback_truncate: If True, fall back to sentence_truncate on LLM failure
        preserve_numbers: If True, explicitly preserve numerical data
        context_type: Type of content ("bullet_point", "paragraph", "title")

    Returns:
        Condensed text that fits within max_chars
    """
    if len(text) <= max_chars:
        return text

    try:
        from backend.services.llm import EnhancedLLMFactory

        llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="content_generation",
            user_id=None,
        )

        # Extract numerical data to preserve
        numerical_data = []
        if preserve_numbers:
            # Find all numbers, percentages, currency values
            number_patterns = [
                r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|M|B|K))?',  # Currency
                r'[\d,]+(?:\.\d+)?%',  # Percentages
                r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b',  # Large numbers with commas
                r'\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|M|B|K)\b',  # Numbers with magnitude
                r'\b(?:Q[1-4]|FY)\s*\d{2,4}\b',  # Fiscal quarters/years
                r'\b\d{4}\b',  # Years
            ]
            for pattern in number_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                numerical_data.extend(matches)

        # Build context-specific instructions
        context_instructions = {
            "bullet_point": "Keep it punchy and actionable. Start with a verb when possible.",
            "paragraph": "Maintain readability and flow. Keep complete sentences.",
            "title": "Be concise but descriptive. Capture the main theme.",
        }
        context_hint = context_instructions.get(context_type, context_instructions["bullet_point"])

        # Build preservation instructions
        preserve_instructions = ""
        if numerical_data:
            unique_numbers = list(set(numerical_data))[:10]  # Limit to first 10 unique
            preserve_instructions = f"""
CRITICAL: You MUST preserve these exact numerical values in your condensed version:
{', '.join(unique_numbers)}

Do not round, approximate, or omit any of these numbers."""

        prompt = f"""Condense this text to under {max_chars} characters while preserving the key meaning.

RULES:
1. Preserve ALL numbers, percentages, and statistics exactly as written
2. Keep the main point and most important facts
3. Use concise, professional language
4. {context_hint}
5. Do NOT add any prefixes like "Here is..." or "The condensed version is..."
6. Output ONLY the condensed text - nothing else
{preserve_instructions}

Original ({len(text)} chars): {text}

Condensed version (under {max_chars} chars):"""

        response = await llm.ainvoke(prompt)
        condensed = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        # Clean up any accidental prefixes the LLM might add
        prefixes_to_remove = [
            "Here is the condensed version:",
            "Condensed version:",
            "Here's the condensed text:",
            "The condensed text is:",
        ]
        for prefix in prefixes_to_remove:
            if condensed.lower().startswith(prefix.lower()):
                condensed = condensed[len(prefix):].strip()

        # Verify it fits
        if len(condensed) <= max_chars:
            # Verify numerical data was preserved (warning only)
            if preserve_numbers and numerical_data:
                preserved_count = sum(1 for num in numerical_data if num in condensed)
                if preserved_count < len(numerical_data) * 0.5:  # Less than 50% preserved
                    logger.debug(
                        "Smart condense: some numerical data may have been lost",
                        original_numbers=len(numerical_data),
                        preserved=preserved_count,
                    )
            return condensed

        # If still too long, truncate the LLM output
        return sentence_truncate(condensed, max_chars)

    except Exception as e:
        logger.warning(f"LLM condense failed, using fallback: {e}")
        if fallback_truncate:
            return sentence_truncate(text, max_chars)
        return text[:max_chars]


async def smart_condense_content(
    content: str,
    max_length: int,
    content_type: str = "bullet_point",
    preserve_numbers: bool = True,
) -> str:
    """Smart content condensation for document generation.

    High-level wrapper around llm_condense_text that provides
    content-type-specific condensation with fallback strategies.

    Args:
        content: Content to condense
        max_length: Maximum character length
        content_type: Type of content (bullet_point, paragraph, title, subtitle)
        preserve_numbers: Whether to preserve numerical data

    Returns:
        Condensed content fitting within max_length
    """
    if len(content) <= max_length:
        return content

    # Map content types to condensation strategies
    type_mapping = {
        "bullet_point": "bullet_point",
        "paragraph": "paragraph",
        "title": "title",
        "subtitle": "title",
        "heading": "title",
        "body": "paragraph",
    }
    context_type = type_mapping.get(content_type, "bullet_point")

    return await llm_condense_text(
        text=content,
        max_chars=max_length,
        fallback_truncate=True,
        preserve_numbers=preserve_numbers,
        context_type=context_type,
    )


# =============================================================================
# Spelling Check
# =============================================================================

def check_spelling(text: str, language: str = "en") -> List[str]:
    """Check spelling in text and return list of misspelled words.

    Args:
        text: Text to check
        language: Language code (en, de, es, fr, etc.)

    Returns:
        List of potentially misspelled words
    """
    try:
        # Try to use enchant if available
        import enchant

        d = enchant.Dict(language)
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        misspelled = [word for word in words if not d.check(word)]
        return misspelled
    except ImportError:
        # Enchant not installed, return empty list
        logger.debug("Enchant not installed, skipping spell check")
        return []
    except Exception as e:
        logger.warning(f"Spell check failed: {e}")
        return []


# =============================================================================
# Color Utilities
# =============================================================================

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color."""
    return f"#{r:02x}{g:02x}{b:02x}"


def get_contrasting_color(hex_color: str) -> str:
    """Get a contrasting color (black or white) for text readability."""
    r, g, b = hex_to_rgb(hex_color)
    # Calculate luminance
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#FFFFFF" if luminance < 0.5 else "#000000"


def lighten_color(hex_color: str, factor: float = 0.3) -> str:
    """Lighten a color by a factor (0-1)."""
    r, g, b = hex_to_rgb(hex_color)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return rgb_to_hex(r, g, b)


def darken_color(hex_color: str, factor: float = 0.3) -> str:
    """Darken a color by a factor (0-1)."""
    r, g, b = hex_to_rgb(hex_color)
    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))
    return rgb_to_hex(r, g, b)


# =============================================================================
# Filename Utilities
# =============================================================================

def sanitize_filename(title: str, max_length: int = 50) -> str:
    """Sanitize title for use as filename.

    Removes/replaces invalid characters for cross-platform compatibility.
    Handles special characters like /, :, |, ?, * that cause issues on various OS.

    Args:
        title: Document title to sanitize
        max_length: Maximum filename length (default 50)

    Returns:
        Safe filename string with only alphanumeric, underscores, and hyphens
    """
    import unicodedata
    # Normalize unicode characters (e.g., convert é to e)
    title = unicodedata.normalize('NFKD', title)
    title = title.encode('ascii', 'ignore').decode('ascii')
    # Keep only alphanumeric, spaces, hyphens, underscores
    title = re.sub(r'[^\w\s-]', '', title)
    # Replace spaces and multiple hyphens/underscores with single underscore
    title = re.sub(r'[-\s]+', '_', title)
    # Remove leading/trailing underscores
    title = title.strip('_')
    # Limit length
    return title[:max_length] if title else 'document'


# =============================================================================
# Theme Color Utilities
# =============================================================================

def get_theme_colors(theme_key: str = "business", custom_colors: dict = None) -> dict:
    """Get theme colors, with fallback to business theme.

    If custom_colors is provided, those values override the theme colors.
    Custom colors can include: primary, secondary, accent, text, background.

    Args:
        theme_key: Key from THEMES dict (business, creative, academic, etc.)
        custom_colors: Optional dict with color overrides

    Returns:
        Dict with primary, secondary, accent, text, background colors
    """
    from .config import THEMES

    theme = THEMES.get(theme_key, THEMES["business"]).copy()

    # Apply custom color overrides if provided
    if custom_colors:
        if "primary" in custom_colors:
            theme["primary"] = custom_colors["primary"]
        if "secondary" in custom_colors:
            theme["secondary"] = custom_colors["secondary"]
        if "accent" in custom_colors:
            theme["accent"] = custom_colors["accent"]
        if "text" in custom_colors:
            theme["text"] = custom_colors["text"]
        if "background" in custom_colors:
            theme["background"] = custom_colors["background"]

    return theme
