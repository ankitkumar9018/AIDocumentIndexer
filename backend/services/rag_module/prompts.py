"""
AIDocumentIndexer - RAG Prompts and Language Configuration
============================================================

System prompts, templates, and language utilities for RAG service.
Supports multilingual document retrieval and response generation.
"""

from typing import List, Tuple


# =============================================================================
# System Prompts
# =============================================================================

RAG_SYSTEM_PROMPT = """You are an intelligent assistant for the AI Document Indexer system.
Your role is to help users find information from their document archive and answer questions based on the retrieved content.

Guidelines:
1. Answer questions based primarily on the provided context from the documents
2. If the context doesn't contain relevant information, say so clearly
3. Always cite your sources by mentioning the document names
4. Be concise but thorough in your responses
5. If asked about something outside the document context, clarify that your knowledge comes from the indexed documents

Remember: You are helping users explore their historical document archive spanning many years of work."""

RAG_PROMPT_TEMPLATE = """Use the following context from the document archive to answer the user's question.
If the context doesn't contain relevant information to answer the question, say so.

Context:
{context}

Question: {question}

Provide a helpful, accurate answer based on the context. Cite specific documents when referencing information.

At the end of your response, on a new line, suggest 2-3 related follow-up questions the user might want to ask, prefixed with "SUGGESTED_QUESTIONS:" and separated by "|". Example:
SUGGESTED_QUESTIONS: What are the key benefits?|How does this compare to alternatives?|When was this implemented?"""

CONVERSATIONAL_RAG_TEMPLATE = """You are having a conversation with a user about their document archive.
Use the retrieved context and conversation history to provide helpful answers.

Retrieved Context:
{context}

Based on this context and our conversation, please answer the user's latest question.
If the context doesn't contain relevant information, acknowledge that and provide what help you can.

At the end of your response, on a new line, suggest 2-3 related follow-up questions the user might want to ask, prefixed with "SUGGESTED_QUESTIONS:" and separated by "|". Example:
SUGGESTED_QUESTIONS: What are the key benefits?|How does this compare to alternatives?|When was this implemented?"""


# =============================================================================
# Language Support
# =============================================================================

# Language code to name mapping for multilingual support
LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
}


def get_language_instruction(language: str, auto_detect: bool = False) -> str:
    """
    Get language instruction for the LLM prompt.

    The system supports:
    - Documents in ANY language (German, French, Chinese, etc.)
    - Queries in ANY language
    - Responses in the user's selected output language OR the query language (auto_detect)

    Args:
        language: Language code for OUTPUT (en, de, es, etc.)
        auto_detect: If True and language is "en", respond in the same language as the question

    Returns:
        Language instruction string
    """
    if language == "en" and auto_detect:
        # Auto-detect mode: respond in the same language as the question
        # Make it VERY explicit for smaller models that don't follow instructions well
        return """
CRITICAL LANGUAGE AND SCRIPT REQUIREMENT (MUST FOLLOW):
1. FIRST, identify the language AND SCRIPT of the USER'S QUESTION (not the documents!)
2. THEN, respond ONLY in that SAME language AND SAME SCRIPT as the user's question
3. IMPORTANT about Indian languages:
   - Hinglish = Hindi written in LATIN/ROMAN script (like "kya hai", "accha hai") - respond in Latin script
   - If user writes in Devanagari (Hindi script like क्या), respond in Devanagari
   - NEVER respond in Gujarati, Bengali, Tamil, or other scripts unless user used them!
4. If the user asks in English, respond in English
5. If the user asks in German, respond in German
6. IGNORE the language of source documents - they may be in any language
7. TRANSLATE all information FROM documents INTO the user's question language AND script

Example: "kya marketing ke baare mea hai?" - Hinglish (Latin script), respond like: "Haan, marketing ke baare mein..."
"""

    if language == "en":
        return ""

    language_name = LANGUAGE_NAMES.get(language, "English")
    return f"""
LANGUAGE REQUIREMENT:
- Your response must be ENTIRELY in {language_name}
- The source documents may be in ANY language (German, English, French, Chinese, etc.)
- Translate and synthesize information from ALL source documents into {language_name}
- The user's question may also be in any language - understand it and respond in {language_name}
- Do NOT mix languages in your response - use only {language_name}
- Technical terms and proper nouns may remain in their original form if commonly used that way
"""


def parse_suggested_questions(content: str) -> Tuple[str, List[str]]:
    """
    Parse suggested questions from response content.

    Args:
        content: Full response content from LLM

    Returns:
        Tuple of (cleaned_content, list_of_suggested_questions)
    """
    suggested_questions = []
    cleaned_content = content

    # Look for SUGGESTED_QUESTIONS: line
    if "SUGGESTED_QUESTIONS:" in content:
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            if line.strip().startswith("SUGGESTED_QUESTIONS:"):
                # Extract questions from this line
                questions_part = line.split("SUGGESTED_QUESTIONS:", 1)[1].strip()
                suggested_questions = [q.strip() for q in questions_part.split("|") if q.strip()]
            else:
                new_lines.append(line)
        cleaned_content = "\n".join(new_lines).rstrip()

    return cleaned_content, suggested_questions
