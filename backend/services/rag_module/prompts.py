"""
AIDocumentIndexer - RAG Prompts and Language Configuration
============================================================

System prompts, templates, and language utilities for RAG service.
Supports multilingual document retrieval and response generation.
Enhanced with few-shot examples, format-specific templates, and small model optimization.
"""

from typing import List, Tuple, Optional


# =============================================================================
# System Prompts - Enhanced with Few-Shot Examples
# =============================================================================

RAG_SYSTEM_PROMPT = """You are an intelligent assistant for document analysis.

## Your Capabilities
- Answer questions based on retrieved document context
- Cite sources with document names and page numbers
- Synthesize information across multiple documents
- Acknowledge when information is incomplete or unavailable

## Response Guidelines
1. Lead with the direct answer in the first sentence
2. Support with evidence from documents
3. Cite sources inline: "According to [Document Name, p.X]..." or "[Document Name]"
4. Note any conflicting information between sources
5. Be specific - use numbers, dates, names when available in context
6. If context lacks information, say so clearly rather than guessing

## Example Response
User: What were the Q3 sales figures?
Assistant: Q3 2024 sales reached $4.2M, a 15% increase from Q2 [Quarterly Report, p.12].

Key highlights:
- North America: $2.1M (+18%)
- Europe: $1.4M (+12%)
- Asia-Pacific: $0.7M (+8%)

The Financial Summary notes this exceeded projections by 5%.

SUGGESTED_QUESTIONS: What drove the North America growth?|How do Q3 results compare to last year?|What are Q4 projections?

Remember: You are helping users explore their document archive. Base your answers on the provided context."""

# Original prompt for backward compatibility
RAG_SYSTEM_PROMPT_LEGACY = """You are an intelligent assistant for the AI Document Indexer system.
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
# Format-Specific Templates (Query Intent Based)
# =============================================================================

COMPARISON_TEMPLATE = """Compare based on the documents provided below.

Context:
{context}

Question: {question}

Instructions:
1. Identify the items/concepts being compared
2. Create a structured comparison with relevant aspects
3. Use a table format if comparing 3+ items
4. Cite sources for each data point
5. Note any gaps where data is missing for some items

Format your response with clear structure. End with suggested follow-up questions.

SUGGESTED_QUESTIONS: [suggest 2-3 related questions separated by |]"""

SUMMARY_TEMPLATE = """Summarize the key points from the documents below.

Context:
{context}

Question: {question}

Provide your response in this structure:
1. **Executive Summary** (2-3 sentences capturing the main message)
2. **Key Points** (bullet list of 3-5 most important findings)
3. **Notable Details** (additional context with citations)

Cite sources for each point. End with suggested follow-up questions.

SUGGESTED_QUESTIONS: [suggest 2-3 related questions separated by |]"""

LIST_TEMPLATE = """Extract and list all items matching the query from the documents below.

Context:
{context}

Question: {question}

Instructions:
1. Format as a numbered list
2. Include source citation for each item: [Document Name]
3. Group related items if applicable
4. Note if the list may be incomplete based on available documents

SUGGESTED_QUESTIONS: [suggest 2-3 related questions separated by |]"""

ANALYTICAL_TEMPLATE = """Analyze and explain based on the documents below.

Context:
{context}

Question: {question}

Think through this systematically:
1. **What is being asked**: Identify the core question
2. **Relevant Evidence**: What facts from the context address this?
3. **Analysis**: How do these facts connect to answer the question?
4. **Conclusion**: Clear answer with supporting citations

If the context lacks sufficient information for a complete analysis, acknowledge this.

SUGGESTED_QUESTIONS: [suggest 2-3 related questions separated by |]"""

TEMPORAL_TEMPLATE = """Provide timeline/historical information based on the documents below.

Context:
{context}

Question: {question}

Instructions:
1. Organize information chronologically when dates are available
2. Use a timeline format: **[Date/Period]**: Event/Change [Source]
3. Note the sequence of events and any causal relationships
4. Highlight key milestones or turning points

If dates are unclear or missing, note this and organize by logical sequence.

SUGGESTED_QUESTIONS: [suggest 2-3 related questions separated by |]"""


# =============================================================================
# Chain-of-Thought Template for Complex Queries
# =============================================================================

COT_TEMPLATE = """Answer the following question step by step based on the provided context.

Context:
{context}

Question: {question}

Think through this systematically:

**Step 1 - Understanding**: What exactly is being asked?

**Step 2 - Evidence Gathering**: What relevant facts are in the context?

**Step 3 - Reasoning**: How do these facts connect to answer the question?

**Step 4 - Conclusion**: State your answer with citations.

ANSWER:
[Your final answer based on the reasoning above, with source citations]

SUGGESTED_QUESTIONS: [suggest 2-3 related questions separated by |]"""


# =============================================================================
# Small Model Optimization Prompt
# =============================================================================

SMALL_MODEL_SYSTEM_PROMPT = """You are a document assistant. Follow these rules strictly:

ALWAYS DO:
- Start with the direct answer in the first sentence
- Use bullet points for multiple items
- Quote exact text with "quotes" when citing important phrases
- Cite sources as [Document Name]
- End with SUGGESTED_QUESTIONS: q1|q2|q3

NEVER DO:
- Make up information not in the provided context
- Give vague responses like "it depends" without specifics
- Skip source citations
- Respond without checking the context first

You will receive context from documents. Answer questions based ONLY on that context.
If the context doesn't contain the answer, say "The provided documents don't contain this information."
"""

SMALL_MODEL_TEMPLATE = """Context from documents:
{context}

Question: {question}

Remember:
1. Answer from context ONLY
2. Cite [Document Name] for each fact
3. Be specific with numbers/dates/names
4. If not in context, say so

Your answer:"""


# =============================================================================
# Tiny Model Optimization (0.5B-3B parameters)
# =============================================================================
# Research: Qwen2.5, Gemma 2B, Phi-2, Llama 3.2 1B/3B need ultra-explicit structure
# - Fixed output format reduces hallucination
# - Short context (200-400 tokens optimal)
# - One task per prompt
# - Temperature 0.1-0.4 for factual tasks (lower = less hallucination)
# - Explicit "don't make up" instructions critical for small models
# - Llama 3.2 1B/3B: trained via knowledge distillation from 8B/70B, good at RAG

TINY_MODEL_SYSTEM_PROMPT = """You answer questions from documents.

OUTPUT FORMAT (copy exactly):
ANSWER: [your 1-2 sentence answer]
SOURCE: [document name]
CONFIDENCE: [high/medium/low]

RULES:
1. Only use information from the provided context
2. If not in context, say "Not found in documents"
3. Keep answers under 50 words
4. Always include SOURCE
5. If you don't know, don't make it up - say so"""

TINY_MODEL_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

OUTPUT FORMAT:
ANSWER: [answer here]
SOURCE: [document name]
CONFIDENCE: [high/medium/low]"""

# =============================================================================
# Llama 3.2 Specific Optimization (1B, 3B models)
# =============================================================================
# Research from Meta's documentation:
# - Llama 3.2 1B/3B trained via pruning + knowledge distillation from 8B/70B
# - Optimized for multilingual dialogue, agentic retrieval, summarization
# - Uses special tokens: <|begin_of_text|>, <|eot_id|>, etc.
# - Supports 128K context but performs best with focused context
# - Lower temperature (0.1-0.4) significantly reduces hallucinations
# - Explicit "don't share false information" instruction from Llama 2 docs

LLAMA_SMALL_SYSTEM_PROMPT = """You are a helpful document assistant.

CRITICAL RULES:
1. Answer ONLY from the provided context
2. If you don't know the answer, say "I don't have this information in the documents"
3. NEVER make up or guess information - this is very important
4. Always cite the document name for each fact
5. Be concise - prefer short, direct answers
6. Think step-by-step before answering complex questions

OUTPUT FORMAT:
- Start with your direct answer
- Include [Document Name] citation for each fact
- End with SUGGESTED_QUESTIONS: q1|q2|q3"""

LLAMA_SMALL_TEMPLATE = """Context from documents:
{context}

Question: {question}

Think step-by-step:
1. What information in the context answers this?
2. Which document(s) contain this information?
3. Am I certain about this, or should I say I don't know?

Your answer (cite sources, be specific):"""

# Patterns to detect tiny models (0.5B-3B parameters)
TINY_MODEL_PATTERNS = [
    "0.5b", "1b", "1.5b", "2b", "3b",
    "qwen2.5-0.5", "qwen2.5-1.5", "qwen2-0.5", "qwen2-1.5",
    "gemma-2b", "gemma2-2b", "gemma:2b",
    "phi-2", "phi2", "phi-1", "phi1",
    "tinyllama", "tiny-llama",
    "stablelm-2", "stablelm2",
    "pythia", "cerebras", "opt-1.3b", "opt-2.7b",
    "bloom-1b", "bloom-3b",
    # Llama 3.2 small models (1B, 3B)
    "llama-3.2-1b", "llama-3.2-3b", "llama3.2-1b", "llama3.2-3b",
    "llama3.2:1b", "llama3.2:3b", "llama-1b", "llama-3b",
    "llama3:1b", "llama3:3b",
]


def is_tiny_model(model_name: str) -> bool:
    """
    Detect if model is a tiny (<3B parameter) model.

    Tiny models need ultra-explicit structure and shorter context
    to produce quality output.

    Args:
        model_name: Name/ID of the model

    Returns:
        True if model is likely <3B parameters
    """
    if not model_name:
        return False

    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in TINY_MODEL_PATTERNS)


# Patterns to detect Llama models specifically
LLAMA_MODEL_PATTERNS = [
    "llama", "llama2", "llama3", "llama-2", "llama-3",
    "llama3.1", "llama3.2", "llama-3.1", "llama-3.2",
    "codellama", "code-llama",
]


def is_llama_model(model_name: str) -> bool:
    """
    Detect if model is a Llama model (any size).

    Llama models benefit from specific prompting patterns including
    chain-of-thought and explicit "don't hallucinate" instructions.

    Args:
        model_name: Name/ID of the model

    Returns:
        True if model is a Llama variant
    """
    if not model_name:
        return False

    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in LLAMA_MODEL_PATTERNS)


def is_llama_small(model_name: str) -> bool:
    """
    Detect if model is a small Llama model (1B-8B).

    These models benefit from step-by-step reasoning prompts
    and explicit anti-hallucination instructions.

    Args:
        model_name: Name/ID of the model

    Returns:
        True if model is a small Llama variant
    """
    if not model_name:
        return False

    model_lower = model_name.lower()

    # Check if it's a Llama model first
    if not is_llama_model(model_name):
        return False

    # Check for small size indicators
    small_indicators = ["1b", "3b", "7b", "8b", ":1b", ":3b", ":7b", ":8b"]
    return any(ind in model_lower for ind in small_indicators)


def get_recommended_temperature(model_name: Optional[str] = None) -> float:
    """
    Get recommended temperature setting based on model.

    Research shows:
    - Tiny models (<3B): 0.1-0.3 for factual tasks
    - Small models (7B-13B): 0.3-0.5
    - Large models (70B+): 0.5-0.7

    Lower temperature reduces hallucinations but may reduce creativity.

    Args:
        model_name: Name of the model being used

    Returns:
        Recommended temperature value (0.0-1.0)
    """
    if not model_name:
        return 0.7  # Default for unknown models

    if is_tiny_model(model_name):
        return 0.2  # Very low for tiny models to minimize hallucination

    if is_llama_small(model_name):
        return 0.3  # Low for small Llama models

    model_lower = model_name.lower()

    # Small models (7B-13B)
    if any(s in model_lower for s in ["7b", "8b", "9b", "13b"]):
        return 0.4

    # Large models
    return 0.7


# =============================================================================
# Template Selection Helper
# =============================================================================

def get_template_for_intent(intent: str, use_cot: bool = False) -> str:
    """
    Get the appropriate prompt template based on query intent.

    Args:
        intent: Query intent (factual, comparison, summary, list, analytical, temporal)
        use_cot: Whether to use chain-of-thought template for complex reasoning

    Returns:
        Appropriate prompt template string
    """
    if use_cot and intent in ("analytical", "comparison"):
        return COT_TEMPLATE

    templates = {
        "comparison": COMPARISON_TEMPLATE,
        "summary": SUMMARY_TEMPLATE,
        "list": LIST_TEMPLATE,
        "analytical": ANALYTICAL_TEMPLATE,
        "temporal": TEMPORAL_TEMPLATE,
        "factual": RAG_PROMPT_TEMPLATE,
    }

    return templates.get(intent, RAG_PROMPT_TEMPLATE)


def get_system_prompt_for_model(model_name: Optional[str] = None) -> str:
    """
    Get appropriate system prompt based on model size/capability.

    Model tiers:
    - Tiny (0.5B-3B): Ultra-explicit structure, fixed output format
    - Llama Small (1B-8B): Step-by-step reasoning, anti-hallucination
    - Small (7B-13B): Explicit instructions, bullet points
    - Large (70B+): Full RAG prompt with few-shot examples

    Args:
        model_name: Name of the model being used

    Returns:
        Appropriate system prompt
    """
    if not model_name:
        return RAG_SYSTEM_PROMPT

    # Check for tiny models first (<3B) - need most explicit instructions
    # But check Llama separately since they have optimized prompts
    if is_tiny_model(model_name):
        # Use Llama-specific prompt for Llama 3.2 1B/3B
        if is_llama_model(model_name):
            return LLAMA_SMALL_SYSTEM_PROMPT
        return TINY_MODEL_SYSTEM_PROMPT

    # Small Llama models (7B-8B) benefit from step-by-step prompts
    if is_llama_small(model_name):
        return LLAMA_SMALL_SYSTEM_PROMPT

    # Small models (7B-13B) that benefit from more explicit instructions
    small_models = [
        "llama", "mistral", "gemma", "phi", "qwen",
        "7b", "8b", "9b", "13b", "mixtral",
    ]

    model_lower = model_name.lower()
    if any(sm in model_lower for sm in small_models):
        return SMALL_MODEL_SYSTEM_PROMPT

    return RAG_SYSTEM_PROMPT


def get_template_for_model(model_name: Optional[str] = None) -> str:
    """
    Get appropriate prompt template based on model size.

    Args:
        model_name: Name of the model being used

    Returns:
        Appropriate prompt template
    """
    if not model_name:
        return RAG_PROMPT_TEMPLATE

    # Tiny models need ultra-structured output
    if is_tiny_model(model_name):
        # Llama 3.2 1B/3B uses its own template with step-by-step
        if is_llama_model(model_name):
            return LLAMA_SMALL_TEMPLATE
        return TINY_MODEL_TEMPLATE

    # Small Llama models (7B-8B) benefit from step-by-step template
    if is_llama_small(model_name):
        return LLAMA_SMALL_TEMPLATE

    model_lower = model_name.lower()
    small_models = ["llama", "mistral", "gemma", "phi", "qwen", "7b", "8b", "9b", "13b", "mixtral"]
    if any(sm in model_lower for sm in small_models):
        return SMALL_MODEL_TEMPLATE

    return RAG_PROMPT_TEMPLATE


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
