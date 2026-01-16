"""
Style Learning and Application Module

Provides functionality for learning writing style from existing documents
and applying learned styles to generation prompts.
"""

import re
from typing import Optional, List

import structlog

from .models import StyleProfile

logger = structlog.get_logger(__name__)


async def learn_style_from_documents(
    document_contents: List[str],
    document_names: Optional[List[str]] = None,
    use_llm_analysis: bool = True,
) -> StyleProfile:
    """Learn writing style from a collection of documents.

    Analyzes document content to extract comprehensive style patterns
    that can be applied to generate consistently styled new documents.

    Args:
        document_contents: List of document text contents
        document_names: Optional list of source document names
        use_llm_analysis: Whether to use LLM for deeper style analysis

    Returns:
        StyleProfile with learned style characteristics
    """
    if not document_contents:
        return StyleProfile()

    # Statistical analysis of documents
    all_sentences = []
    all_paragraphs = []
    word_counts = []
    total_words = 0
    uses_bullets = False
    uses_numbered = False
    uses_bold = False
    uses_italic = False
    first_person_count = 0
    contraction_count = 0
    passive_patterns = 0
    total_sentences = 0

    for content in document_contents:
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        all_sentences.extend(sentences)

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        all_paragraphs.extend(paragraphs)

        # Word count per sentence
        for sentence in sentences:
            words = sentence.split()
            word_counts.append(len(words))
            total_words += len(words)
            total_sentences += 1

            # Check for first person
            if re.search(r'\b(I|we|our|my)\b', sentence, re.IGNORECASE):
                first_person_count += 1

            # Check for contractions
            if re.search(r"'(t|s|re|ll|ve|d)\b|n't\b", sentence, re.IGNORECASE):
                contraction_count += 1

            # Check for passive voice patterns
            if re.search(r'\b(is|are|was|were|been|being)\s+\w+ed\b', sentence, re.IGNORECASE):
                passive_patterns += 1

        # Check for formatting patterns
        if '•' in content or re.search(r'^[-*]\s', content, re.MULTILINE):
            uses_bullets = True
        if re.search(r'^\d+[.)]\s', content, re.MULTILINE):
            uses_numbered = True
        if '**' in content or '<b>' in content.lower():
            uses_bold = True
        if '_' in content or '*' in content or '<i>' in content.lower():
            uses_italic = True

    # Calculate averages
    avg_sentence_length = sum(word_counts) / len(word_counts) if word_counts else 15.0
    sentences_per_para = []
    for para in all_paragraphs:
        para_sentences = re.split(r'[.!?]+', para)
        para_sentences = [s for s in para_sentences if s.strip()]
        sentences_per_para.append(len(para_sentences))
    avg_paragraph_length = sum(sentences_per_para) / len(sentences_per_para) if sentences_per_para else 4.0

    # Calculate ratios
    first_person_ratio = first_person_count / total_sentences if total_sentences > 0 else 0
    contraction_ratio = contraction_count / total_sentences if total_sentences > 0 else 0
    passive_ratio = passive_patterns / total_sentences if total_sentences > 0 else 0

    # Determine vocabulary level based on average word length
    all_words = ' '.join(document_contents).split()
    avg_word_length = sum(len(w) for w in all_words) / len(all_words) if all_words else 5
    if avg_word_length > 7:
        vocabulary_level = "advanced"
    elif avg_word_length > 5.5:
        vocabulary_level = "moderate"
    else:
        vocabulary_level = "simple"

    # Determine structure pattern
    if uses_bullets and not uses_numbered:
        structure_pattern = "bullet-lists"
    elif uses_numbered:
        structure_pattern = "headers-heavy"
    elif avg_paragraph_length > 5:
        structure_pattern = "paragraphs"
    else:
        structure_pattern = "mixed"

    # Determine formality
    if contraction_ratio > 0.2 or first_person_ratio > 0.3:
        formality = "casual"
    elif contraction_ratio < 0.05 and first_person_ratio < 0.1:
        formality = "formal"
    else:
        formality = "moderate"

    # Build initial profile
    profile = StyleProfile(
        vocabulary_level=vocabulary_level,
        formality=formality,
        avg_sentence_length=round(avg_sentence_length, 1),
        avg_paragraph_length=round(avg_paragraph_length, 1),
        structure_pattern=structure_pattern,
        bullet_preference=uses_bullets,
        uses_passive_voice=round(passive_ratio, 2),
        uses_first_person=first_person_ratio > 0.1,
        uses_contractions=contraction_ratio > 0.1,
        uses_bold_emphasis=uses_bold,
        uses_italic_emphasis=uses_italic,
        uses_numbered_lists=uses_numbered,
        source_documents=document_names or [],
        confidence_score=min(1.0, len(document_contents) * 0.2),  # More docs = higher confidence
    )

    # Use LLM for deeper analysis if enabled
    if use_llm_analysis and document_contents:
        try:
            from backend.services.llm import EnhancedLLMFactory

            llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="content_generation",
                user_id=None,
            )

            # Sample text for LLM analysis
            sample_text = "\n\n---\n\n".join([
                content[:1500] for content in document_contents[:5]
            ])

            analysis_prompt = f"""Analyze this text sample and identify writing style characteristics.

TEXT SAMPLES:
{sample_text}

Provide a JSON response with:
{{
    "tone": "formal" | "casual" | "technical" | "friendly" | "academic",
    "key_phrases": ["list of 3-5 common phrases or patterns"],
    "domain_terms": ["list of 3-5 domain-specific terms used"],
    "heading_style": "title_case" | "sentence_case" | "all_caps",
    "recommended_tone": "brief description of how to write in this style"
}}

Return ONLY valid JSON:"""

            response = await llm.ainvoke(analysis_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse LLM response
            import json
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                llm_analysis = json.loads(json_match.group())
                profile.tone = llm_analysis.get("tone", profile.tone)
                profile.key_phrases = llm_analysis.get("key_phrases", [])
                profile.domain_terms = llm_analysis.get("domain_terms", [])
                profile.heading_style = llm_analysis.get("heading_style", "title_case")
                profile.confidence_score = min(1.0, profile.confidence_score + 0.2)

        except Exception as e:
            logger.warning(f"LLM style analysis failed: {e}")

    return profile


def apply_style_to_prompt(prompt: str, style: StyleProfile) -> str:
    """Apply style profile instructions to a generation prompt.

    Args:
        prompt: Original generation prompt
        style: StyleProfile to apply

    Returns:
        Prompt with style instructions appended
    """
    style_instructions = f"""
---WRITING STYLE GUIDANCE (internal - follow but do not output)---
Based on existing documents, match this style:

TONE & FORMALITY:
- Tone: {style.tone}
- Formality: {style.formality}
- Vocabulary: {style.vocabulary_level} complexity

SENTENCE STRUCTURE:
- Target sentence length: ~{style.avg_sentence_length:.0f} words
- Target paragraph length: ~{style.avg_paragraph_length:.0f} sentences
{"- Use first person (I, we) when appropriate" if style.uses_first_person else "- Avoid first person pronouns"}
{"- Contractions are OK (don't, can't)" if style.uses_contractions else "- Avoid contractions (use 'do not', 'cannot')"}

FORMATTING:
- Structure: {style.structure_pattern}
{"- Prefer bullet points for lists" if style.bullet_preference else "- Prefer prose over bullet points"}
{"- Use **bold** for emphasis" if style.uses_bold_emphasis else ""}
{"- Use _italic_ for emphasis" if style.uses_italic_emphasis else ""}
- Heading style: {style.heading_style}
"""

    if style.key_phrases:
        style_instructions += f"\nKEY PHRASES to use when relevant:\n"
        for phrase in style.key_phrases[:5]:
            style_instructions += f"- \"{phrase}\"\n"

    if style.domain_terms:
        style_instructions += f"\nDOMAIN TERMS to incorporate:\n"
        for term in style.domain_terms[:5]:
            style_instructions += f"- {term}\n"

    style_instructions += "---END STYLE GUIDANCE---\n"

    return prompt + "\n" + style_instructions
