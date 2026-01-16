"""
AIDocumentIndexer - Document Generation Utilities
===================================================

Text processing utilities for document generation: markdown stripping,
text truncation, content condensation, and filename sanitization.
"""

import re
import unicodedata
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


def strip_markdown(text: str) -> str:
    """Remove markdown formatting, returning clean text.

    Used by PPTX, XLSX, and other generators that don't support markdown.
    """
    if not text:
        return ""
    # Remove headers (# ## ### #### etc.)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
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
        r'^.*?[Ww]e (?:will|\'ll) (?:track|provide|create|use|measure).*?:\s*\n?',
        r'^.*?[Tt]o (?:measure|track|monitor) (?:the )?success.*?:\s*\n?',
        # Style instructions appearing as content
        r'^\s*[-•*▪◦▸]?\s*(?:Writing )?[Ss]tyle [Rr]equirements?.*$',
        r'^\s*[-•*▪◦▸]?\s*[Mm]aintain a professional tone.*$',
        r'^\s*[-•*▪◦▸]?\s*[Uu]se (?:simple|clear|concise) language.*$',
        r'^\s*[-•*▪◦▸]?\s*[Tt]he (?:new )?content should (?:use|have|be|match).*$',
        r'^\s*[-•*▪◦▸]?\s*[Kk]ey characteristics of the desired writing style.*$',
        r'^\s*[-•*▪◦▸]?\s*[Uu]sing medium-length sentences.*$',
        r'^\s*[-•*▪◦▸]?\s*[Ii]ncorporating action verbs.*$',
        r'^\s*[-•*▪◦▸]?\s*[Bb]y following these style requirements.*$',
        # Preamble "To [action]..." patterns
        r'^\s*[-•*▪◦▸]?\s*[Tt]o (?:create|establish|build|develop|implement|ensure|achieve) (?:a |the )?(?:buzz|strong|solid|effective|successful).*?,\s*we\s+.*$',
        r'^\s*[-•*▪◦▸]?\s*[Tt]o (?:create|establish|build|develop|implement|ensure|achieve).*?:\s*$',
        r'^\s*[-•*▪◦▸]?\s*[Ii]n order to (?:create|establish|build|develop|implement|ensure|achieve).*?:\s*$',
        r'^\s*[-•*▪◦▸]?\s*[Ff]or the purpose of (?:creating|establishing|building|developing).*?:\s*$',
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
    abbreviations = r'(?<![Mm]r)(?<![Mm]rs)(?<![Mm]s)(?<![Dd]r)(?<![Pp]rof)(?<![Ii]nc)(?<![Ll]td)(?<![Jj]r)(?<![Ss]r)(?<![Vv]s)(?<!etc)(?<!e\.g)(?<!i\.e)'

    # Find sentence endings: period/exclamation/question followed by space or end
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
            unique_numbers = list(set(numerical_data))[:10]
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

        # Clean up any accidental prefixes
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
            if preserve_numbers and numerical_data:
                preserved_count = sum(1 for num in numerical_data if num in condensed)
                if preserved_count < len(numerical_data) * 0.5:
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


def check_spelling(text: str) -> dict:
    """Check spelling and return suggestions for review.

    This flags potential spelling issues for user approval rather than
    auto-correcting, allowing users to decide what to fix.

    Returns:
        dict with:
        - has_issues: bool indicating if spelling issues were found
        - issues: list of dicts with word, position, suggestion, context
        - original: the original text
    """
    if not text:
        return {"has_issues": False, "issues": [], "original": text}

    try:
        from spellchecker import SpellChecker

        spell = SpellChecker()
        words = text.split()
        issues = []

        for i, word in enumerate(words):
            # Clean punctuation from word
            clean_word = word.strip('.,!?:;"\'-()[]{}')
            if not clean_word or len(clean_word) < 3:
                continue

            # Skip common patterns that aren't words
            if clean_word.isdigit():
                continue
            if any(c.isdigit() for c in clean_word):  # Skip alphanumeric codes
                continue
            if clean_word.startswith(('@', '#', 'http', 'www')):
                continue

            # Check if word is known
            if clean_word.lower() not in spell:
                correction = spell.correction(clean_word.lower())
                # Only suggest if we have a different correction
                if correction and correction != clean_word.lower():
                    # Get context (surrounding words)
                    context_start = max(0, i - 2)
                    context_end = min(len(words), i + 3)
                    context = ' '.join(words[context_start:context_end])

                    issues.append({
                        "word": clean_word,
                        "position": i,
                        "suggestion": correction,
                        "context": context
                    })

        return {
            "has_issues": len(issues) > 0,
            "issues": issues[:20],  # Limit to top 20 issues
            "original": text
        }
    except ImportError:
        logger.warning("pyspellchecker not installed, skipping spell check")
        return {"has_issues": False, "issues": [], "original": text}
    except Exception as e:
        logger.warning(f"Spell check failed: {e}")
        return {"has_issues": False, "issues": [], "original": text}
