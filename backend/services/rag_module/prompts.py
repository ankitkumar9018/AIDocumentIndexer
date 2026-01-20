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
# - Weak models (1B, Llama2) benefit from few-shot examples

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

# Few-shot system prompt for very weak models (Llama 3.2 1B, Llama2 7B)
# Research shows these models benefit significantly from in-context examples
LLAMA_WEAK_SYSTEM_PROMPT = """You are a helpful document assistant.

CRITICAL RULES:
1. Answer ONLY from the provided context
2. If you don't know the answer, say "I don't have this information in the documents"
3. NEVER make up or guess information - this is very important
4. Always cite the document name for each fact
5. Be concise - prefer short, direct answers

Here's an example of how to answer:

EXAMPLE:
Context: "The quarterly revenue was $5.2M according to the Q3 Financial Report. The CEO mentioned in the Annual Meeting Notes that this represents 15% growth."

Question: "What was the quarterly revenue?"

Step-by-step:
1. Look for revenue information in context → Found "$5.2M"
2. Which document? → "Q3 Financial Report"
3. Am I certain? → Yes, explicitly stated

Answer: The quarterly revenue was $5.2M [Q3 Financial Report]. This represented 15% growth according to the CEO [Annual Meeting Notes].

SUGGESTED_QUESTIONS: What was the growth rate?|Who reported this revenue?|Which quarter was this?"""

LLAMA_SMALL_TEMPLATE = """Context from documents:
{context}

Question: {question}

Think step-by-step:
1. What information in the context answers this?
2. Which document(s) contain this information?
3. Am I certain about this, or should I say I don't know?

Your answer (cite sources, be specific):"""


# =============================================================================
# Qwen Model Optimization (All sizes, especially strong at structured output)
# =============================================================================
# Research: Qwen2.5 excels at JSON/structured output and instruction following
# - Inherits ChatML format: <|im_start|>system, <|im_start|>user, <|im_end|>
# - More resilient to diversity of system prompts
# - Significant improvements in generating structured outputs
# - Better at understanding structured data (tables)
# - No pruning, KD optimization at finetuning stage

QWEN_SMALL_SYSTEM_PROMPT = """You are a document assistant optimized for precise, structured responses.

ALWAYS:
- Start with direct, factual answer
- Cite sources as [Document Name]
- Use bullet points for clarity
- Include SUGGESTED_QUESTIONS: q1|q2|q3 at end

NEVER:
- Make up information not in context
- Provide vague responses
- Skip source citations
- Mix information from different documents without clear attribution

If the context doesn't contain the answer, explicitly state: "The provided documents don't contain this information."
"""

QWEN_SMALL_TEMPLATE = """Context from documents:
{context}

Question: {question}

Provide a clear, well-structured answer based ONLY on the context above. Cite sources for each fact.

Your answer:"""


# =============================================================================
# Phi Model Optimization (Format-sensitive, good at reasoning)
# =============================================================================
# Research: Phi-3 Mini best at reasoning/math in class, but format-sensitive
# - Uses embedded role tags: <|system|>, <|user|>, <|assistant|>, <|end|>
# - Poor formatting negatively impacts instruction adherence
# - Quantization-aware training (4-bit works exceptionally well)

PHI_SMALL_SYSTEM_PROMPT = """You are a document Q&A assistant.

FORMAT RULES (STRICT):
- Start with direct answer
- Cite sources as [Document Name]
- Explain reasoning briefly if needed
- End with SUGGESTED_QUESTIONS: q1|q2|q3

NEVER:
- Skip source citations
- Provide information not in context
- Use vague language without specifics
- Make assumptions beyond what's stated
"""

PHI_SMALL_TEMPLATE = """Context from documents:
{context}

Question: {question}

Instructions:
1. Find relevant information in context
2. Provide clear, direct answer
3. Cite [Document Name] for each fact
4. If not in context, say so

Your answer:"""


# =============================================================================
# Gemma Model Optimization (Calm and safe)
# =============================================================================
# Research: Gemma 2B is "calm and safe" - rarely goes off-track
# - Uses <start_of_turn>user, <start_of_turn>model, <end_of_turn> tags
# - Require strict user/assistant alternation
# - No separate system message (embed in first user message)
# - Good for constrained tasks and safety

GEMMA_SMALL_SYSTEM_PROMPT = """You answer questions from documents. Be calm, accurate, and safe.

ALWAYS:
- Start with direct factual answer
- Cite [Document Name] for each fact
- Say "Not in documents" if unsure
- Be concise and specific

NEVER:
- Guess or make up information
- Provide unsafe or unreliable content
- Skip source attribution
"""

GEMMA_SMALL_TEMPLATE = """Context from documents:
{context}

Question: {question}

Answer based ONLY on the context. Cite sources. Be specific.

Your answer:"""


# =============================================================================
# DeepSeek Model Optimization (R1 distilled versions 1.5B-14B)
# =============================================================================
# Research: DeepSeek-R1 distilled models are strong at reasoning
# - Use special reasoning tokens <think> </think>
# - Good at step-by-step analysis
# - Benefit from clear structure in prompts
# - DeepSeek-R1-Distill-Llama-8B and smaller versions are very capable

DEEPSEEK_SMALL_SYSTEM_PROMPT = """You are a document Q&A assistant optimized for reasoning.

ALWAYS:
- Think through the question step-by-step
- Start with direct, factual answer
- Cite sources as [Document Name]
- Be specific and precise
- Include SUGGESTED_QUESTIONS: q1|q2|q3 at end

NEVER:
- Make up information not in context
- Skip reasoning steps for complex questions
- Provide answers without source citations
- Use information beyond the provided documents

If the context doesn't contain the answer, state: "The documents don't contain this information."
"""

DEEPSEEK_SMALL_TEMPLATE = """Context from documents:
{context}

Question: {question}

Analyze step-by-step:
1. What is being asked?
2. What relevant information is in the context?
3. Which documents contain this information?

Your answer (cite sources):"""


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


# Patterns to detect Qwen models
QWEN_MODEL_PATTERNS = [
    "qwen", "qwen2", "qwen2.5", "qwen-2", "qwen-2.5",
]


def is_qwen_model(model_name: str) -> bool:
    """
    Detect if model is a Qwen model.

    Qwen models (especially Qwen2.5) excel at:
    - Structured output (JSON)
    - Instruction following
    - Understanding structured data (tables)

    Args:
        model_name: Name/ID of the model

    Returns:
        True if model is a Qwen variant
    """
    if not model_name:
        return False

    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in QWEN_MODEL_PATTERNS)


# Patterns to detect Phi models
PHI_MODEL_PATTERNS = [
    "phi", "phi-1", "phi-2", "phi-3", "phi1", "phi2", "phi3",
]


def is_phi_model(model_name: str) -> bool:
    """
    Detect if model is a Phi model.

    Phi models are:
    - Format-sensitive (poor formatting hurts performance)
    - Good at math and reasoning
    - Quantization-aware (4-bit works well)

    Args:
        model_name: Name/ID of the model

    Returns:
        True if model is a Phi variant
    """
    if not model_name:
        return False

    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in PHI_MODEL_PATTERNS)


# Patterns to detect Gemma models
GEMMA_MODEL_PATTERNS = [
    "gemma", "gemma-2", "gemma2", "gemma:2b",
]


def is_gemma_model(model_name: str) -> bool:
    """
    Detect if model is a Gemma model.

    Gemma models are:
    - "Calm and safe" - rarely go off-track
    - Good for constrained tasks
    - Require strict user/assistant alternation

    Args:
        model_name: Name/ID of the model

    Returns:
        True if model is a Gemma variant
    """
    if not model_name:
        return False

    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in GEMMA_MODEL_PATTERNS)


# Patterns to detect DeepSeek models
DEEPSEEK_MODEL_PATTERNS = [
    "deepseek", "deepseek-r1", "deepseek-coder", "deepseek-v3", "deep-seek",
]


def is_deepseek_model(model_name: str) -> bool:
    """
    Detect if model is a DeepSeek model.

    DeepSeek models (especially R1) are:
    - Strong at reasoning and coding
    - Use special reasoning tokens <think> </think>
    - Benefit from structured prompts
    - DeepSeek-R1 distilled versions (1.5B-14B) are very capable

    Args:
        model_name: Name/ID of the model

    Returns:
        True if model is a DeepSeek variant
    """
    if not model_name:
        return False

    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in DEEPSEEK_MODEL_PATTERNS)


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


def is_llama_weak(model_name: str) -> bool:
    """
    Detect if model is a very weak Llama model that benefits from few-shot examples.

    Research shows:
    - Llama 3.2 1B: Weakest model, significant improvement with in-context examples
    - Llama2 7B: Older architecture, benefits from few-shot prompting
    - Llama 3.2 3B and Llama 3.1/3.2 8B: Knowledge-distilled, don't need examples

    Args:
        model_name: Name/ID of the model

    Returns:
        True if model is a weak Llama that needs few-shot examples
    """
    if not model_name:
        return False

    model_lower = model_name.lower()

    # Check if it's a Llama model first
    if not is_llama_model(model_name):
        return False

    # Llama 3.2 1B is the weakest - always needs examples
    if "llama3.2" in model_lower or "llama-3.2" in model_lower:
        if "1b" in model_lower or ":1b" in model_lower:
            return True

    # Llama2 models (any size 7B-70B) benefit from examples due to older architecture
    # They weren't knowledge-distilled like Llama 3.2
    if "llama2" in model_lower or "llama-2" in model_lower:
        return True

    return False


def get_recommended_temperature(model_name: Optional[str] = None) -> float:
    """
    Get recommended temperature setting based on model (research-backed 2026).

    Research findings:
    - Tiny models (<3B): 0.2 optimal for factual RAG tasks
    - Small Llama (7B-8B): 0.3-0.4 significantly reduces hallucinations
    - Qwen models: 0.3 optimal for structured output
    - Small models (7B-13B): 0.4 balances accuracy and coherence
    - Large models (70B+): 0.7 for natural conversation

    Lower temperature dramatically reduces hallucinations in small models.

    Args:
        model_name: Name of the model being used

    Returns:
        Recommended temperature value (0.0-1.0)
    """
    if not model_name:
        return 0.7  # Default for unknown models

    # Tiny models need very low temperature to minimize hallucination
    if is_tiny_model(model_name):
        return 0.2

    # Small Llama models (1B-8B)
    if is_llama_small(model_name):
        return 0.3

    # Llama models without explicit size (e.g., "llama3.2:latest")
    # Most local Llama deployments use smaller variants (3B or less), so default to 0.3
    if is_llama_model(model_name):
        return 0.3

    # Qwen models benefit from lower temp for structured output
    if is_qwen_model(model_name):
        return 0.3

    # DeepSeek models benefit from lower temp for reasoning
    if is_deepseek_model(model_name):
        return 0.3

    # Phi models benefit from lower temp for format adherence
    if is_phi_model(model_name):
        return 0.3

    # Gemma models benefit from lower temp for safety
    if is_gemma_model(model_name):
        return 0.3

    model_lower = model_name.lower()

    # Small models (7B-13B)
    if any(s in model_lower for s in ["7b", "8b", "9b", "13b"]):
        return 0.4

    # Large models
    return 0.7


def get_sampling_config(model_name: Optional[str] = None) -> dict:
    """
    Get comprehensive sampling configuration based on model (research-backed 2026).

    Research shows:
    - Lower temperature reduces hallucinations (most impactful parameter)
    - top_p=0.9 provides good balance for most tasks
    - top_k=50 eliminates low-quality options
    - Best practice: Use temperature + ONE of top_p/top_k, not both
    - Exception: Tiny models benefit from both for extra constraint
    - Repeat penalty (1.1) reduces repetition in tiny models

    Args:
        model_name: Name of the model being used

    Returns:
        Dict with temperature, top_p, top_k, repeat_penalty configuration
    """
    config = {
        "temperature": get_recommended_temperature(model_name),
        "top_p": 0.9,
        "top_k": None,  # Let top_p handle it for most models
        "repeat_penalty": 1.0,  # Default: no penalty
    }

    # Tiny models (<3B) need extra constraints - use both top_p and top_k
    # Also add slight repeat penalty to reduce repetition
    if model_name and is_tiny_model(model_name):
        config["top_k"] = 50
        config["repeat_penalty"] = 1.1  # Slight penalty to reduce repetition

    return config


def optimize_chunk_count_for_model(
    intent_top_k: int,
    model_name: Optional[str],
) -> int:
    """
    Cap chunk count based on model's effective context handling ability.

    Research shows:
    - Tiny models (<3B): Struggle with long context even if they support large context windows
    - Llama 3.2 1B specifically: Weakest model, needs minimal context (5 chunks max)
    - Small models (7B-8B): Can handle moderate context (10 chunks optimal)
    - "Lost in the Middle" problem intensifies with model size

    This prevents context overload while maintaining scalability for thousands of files.
    The retrieval system will get the best N chunks, this just caps N appropriately.

    Args:
        intent_top_k: Suggested top_k from query intent classification
        model_name: Name of the model being used

    Returns:
        Optimized chunk count (capped based on model capability)
    """
    if not model_name:
        return intent_top_k

    model_lower = model_name.lower()

    # Llama 3.2 1B: Weakest model, max 5 chunks
    # This is the most aggressive cap for the smallest, weakest model
    if is_llama_model(model_name) and "1b" in model_lower:
        return min(intent_top_k, 5)

    # Tiny models (<3B): Max 6 chunks regardless of intent
    # Even with 128K context window, tiny models struggle with long context
    if is_tiny_model(model_name):
        return min(intent_top_k, 6)

    # Small models (7B-8B): Max 10 chunks
    # Good balance between context and performance
    if is_llama_small(model_name) or any(s in model_lower for s in ["7b", "8b"]):
        return min(intent_top_k, 10)

    # Larger models can handle full intent-based top_k
    return intent_top_k


def get_adaptive_sampling_config(
    model_name: Optional[str],
    query_intent: Optional[str] = None,
) -> dict:
    """
    Get sampling configuration adapted for both model AND query type.

    Research shows different query types have different hallucination risks:
    - Factual queries: Need ultra-low temperature (deterministic, precise)
    - Navigational queries: Need ultra-low temperature (finding specific items)
    - Exploratory queries: Can use slightly higher temperature (broader, more creative)
    - Comparative queries: Medium temperature (balanced analysis)

    Args:
        model_name: Name of the model being used
        query_intent: Query intent string (factual, exploratory, etc.)

    Returns:
        Sampling configuration optimized for model + intent combination
    """
    # Start with base model configuration
    config = get_sampling_config(model_name)

    if not query_intent or not model_name:
        return config

    # Only adjust temperature for small models (tiny + small)
    # Large models can handle their base temperature across all intents
    is_small = (
        is_tiny_model(model_name)
        or is_llama_small(model_name)
        or any(s in model_name.lower() for s in ["7b", "8b", "9b", "13b"])
    )

    if not is_small:
        return config

    intent_lower = query_intent.lower()

    # Ultra-precise queries: Lower temperature even more
    if intent_lower in ["factual", "navigational"]:
        config["temperature"] = max(0.1, config["temperature"] - 0.1)

    # Exploratory/creative queries: Slightly higher temperature
    elif intent_lower in ["exploratory", "comparative"]:
        config["temperature"] = min(0.5, config["temperature"] + 0.05)

    return config


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
    Get appropriate system prompt based on model family and size (research-backed 2026).

    Model tiers and families:
    - Weak Llama (1B, Llama2): Few-shot examples (significant improvement)
    - Tiny Llama (3B): Step-by-step reasoning (knowledge-distilled from 8B/70B)
    - Tiny Qwen (0.5B-3B): Structured output focus (excels at JSON)
    - Tiny Phi (2B-3.8B): Format-sensitive reasoning
    - Tiny Gemma (2B): Calm and safe responses
    - Tiny Generic (<3B): Ultra-explicit fixed format
    - Small Llama (7B-8B): Step-by-step with anti-hallucination
    - Small Qwen (7B+): Structured output at all sizes
    - Small Phi/Gemma (7B): Family-specific handling
    - Small Generic (7B-13B): Explicit instructions
    - Large (70B+): Full RAG prompt with few-shot examples

    Args:
        model_name: Name of the model being used

    Returns:
        Appropriate system prompt
    """
    if not model_name:
        return RAG_SYSTEM_PROMPT

    # Very weak Llama models (1B, Llama2) - need few-shot examples
    # Research shows these benefit significantly from in-context examples
    if is_llama_weak(model_name):
        return LLAMA_WEAK_SYSTEM_PROMPT

    # Tiny models (<3B) - check family first for specialized prompts
    if is_tiny_model(model_name):
        # Llama 3.2 3B: knowledge-distilled, use step-by-step (not weak, don't need examples)
        if is_llama_model(model_name):
            return LLAMA_SMALL_SYSTEM_PROMPT
        # Qwen tiny models: excel at structured output
        elif is_qwen_model(model_name):
            return QWEN_SMALL_SYSTEM_PROMPT
        # Phi tiny models: format-sensitive
        elif is_phi_model(model_name):
            return PHI_SMALL_SYSTEM_PROMPT
        # Gemma 2B: calm and safe
        elif is_gemma_model(model_name):
            return GEMMA_SMALL_SYSTEM_PROMPT
        # Generic tiny models: ultra-explicit fixed format
        return TINY_MODEL_SYSTEM_PROMPT

    # Small Llama models (7B-8B) - check if Llama2 (needs examples) or Llama3+ (doesn't)
    if is_llama_small(model_name):
        # Llama2 7B needs few-shot (already handled by is_llama_weak above, but double-check)
        # Llama 3.1/3.2 7B-8B don't need examples
        return LLAMA_SMALL_SYSTEM_PROMPT

    # Qwen benefits from structured prompts at all sizes
    if is_qwen_model(model_name):
        return QWEN_SMALL_SYSTEM_PROMPT

    # Phi models - format-sensitive at all sizes
    if is_phi_model(model_name):
        return PHI_SMALL_SYSTEM_PROMPT

    # Gemma models - calm and safe at all sizes
    if is_gemma_model(model_name):
        return GEMMA_SMALL_SYSTEM_PROMPT

    # DeepSeek models - strong at reasoning
    if is_deepseek_model(model_name):
        return DEEPSEEK_SMALL_SYSTEM_PROMPT

    # Generic small models (7B-13B)
    model_lower = model_name.lower()
    if any(s in model_lower for s in ["7b", "8b", "9b", "13b", "14b", "mistral", "mixtral"]):
        return SMALL_MODEL_SYSTEM_PROMPT

    return RAG_SYSTEM_PROMPT


def get_template_for_model(model_name: Optional[str] = None) -> str:
    """
    Get appropriate prompt template based on model family and size (research-backed 2026).

    Args:
        model_name: Name of the model being used

    Returns:
        Appropriate prompt template
    """
    if not model_name:
        return RAG_PROMPT_TEMPLATE

    # Tiny models (<3B) - check family first
    if is_tiny_model(model_name):
        # Llama 3.2 1B/3B: step-by-step reasoning template
        if is_llama_model(model_name):
            return LLAMA_SMALL_TEMPLATE
        # Qwen tiny: structured output template
        elif is_qwen_model(model_name):
            return QWEN_SMALL_TEMPLATE
        # Phi tiny: format-sensitive template
        elif is_phi_model(model_name):
            return PHI_SMALL_TEMPLATE
        # Gemma 2B: simple direct template
        elif is_gemma_model(model_name):
            return GEMMA_SMALL_TEMPLATE
        # Generic tiny: ultra-structured fixed format
        return TINY_MODEL_TEMPLATE

    # Small Llama models (7B-8B) - step-by-step reasoning
    if is_llama_small(model_name):
        return LLAMA_SMALL_TEMPLATE

    # Qwen at all sizes - structured template
    if is_qwen_model(model_name):
        return QWEN_SMALL_TEMPLATE

    # Phi models - format-sensitive template
    if is_phi_model(model_name):
        return PHI_SMALL_TEMPLATE

    # Gemma models - simple template
    if is_gemma_model(model_name):
        return GEMMA_SMALL_TEMPLATE

    # DeepSeek models - reasoning-focused template
    if is_deepseek_model(model_name):
        return DEEPSEEK_SMALL_TEMPLATE

    # Generic small models (7B-13B)
    model_lower = model_name.lower()
    if any(s in model_lower for s in ["7b", "8b", "9b", "13b", "14b", "mistral", "mixtral"]):
        return SMALL_MODEL_TEMPLATE

    return RAG_PROMPT_TEMPLATE


# =============================================================================
# Model-Specific Base Instructions (for Agent Mode)
# =============================================================================

def get_model_base_instructions(model_name: Optional[str] = None) -> str:
    """
    Get model-specific base instructions that can be prepended to user's custom prompts.

    These are lightweight behavioral hints that don't override user intent but help
    small models stay on-track. Use these for agent mode, workflows, and tasks where
    users provide custom prompts.

    **Key Principle**: These are FOUNDATIONAL instructions, not full system prompts.
    User's custom prompts take precedence and define the actual task.

    For RAG queries, use get_system_prompt_for_model() instead (full prompts).

    Args:
        model_name: Name of the model being used

    Returns:
        Model-specific base instructions (empty string for large models)
    """
    if not model_name:
        return ""

    # Large models don't need base instructions - they're capable enough
    model_lower = model_name.lower()
    is_large = not (
        is_tiny_model(model_name)
        or is_llama_small(model_name)
        or any(s in model_lower for s in ["7b", "8b", "9b", "13b", "14b"])
    )

    if is_large:
        return ""

    # Very weak Llama models (1B, Llama2) - need explicit grounding
    if is_llama_weak(model_name):
        return """Core Behavioral Rules:
- Follow instructions precisely as given
- If you don't know something, say so explicitly
- Never make up information or guess
- Be concise and direct in responses
- Cite sources when referencing information

"""

    # Tiny Llama (3B, knowledge-distilled) - step-by-step thinking
    if is_tiny_model(model_name) and is_llama_model(model_name):
        return """Approach:
- Think step-by-step before responding
- Be precise and factual
- Stay focused on the given task
- Cite sources when applicable

"""

    # Qwen models - structured output focus
    if is_qwen_model(model_name):
        return """Output Guidelines:
- Use structured, organized responses
- Be precise and specific
- Follow any format requirements exactly
- Stay factual and grounded

"""

    # Phi models - format-sensitive
    if is_phi_model(model_name):
        return """Format Requirements:
- Follow format instructions precisely
- Be specific and detailed
- Maintain consistency in output structure
- Explain reasoning when needed

"""

    # Gemma models - calm and safe
    if is_gemma_model(model_name):
        return """Response Style:
- Be calm, accurate, and safe
- Provide clear, direct answers
- Stay on-topic and focused
- Avoid speculation or guessing

"""

    # DeepSeek models - reasoning focus
    if is_deepseek_model(model_name):
        return """Reasoning Approach:
- Think through problems step-by-step
- Be analytical and precise
- Show your reasoning when helpful
- Stay grounded in facts

"""

    # Generic tiny models - ultra-explicit
    if is_tiny_model(model_name):
        return """Critical Rules:
- Answer ONLY what is asked
- Never make up information
- Be concise and direct
- Say "I don't know" when uncertain

"""

    # Small models (7B-13B) - gentle guidance
    if any(s in model_lower for s in ["7b", "8b", "9b", "13b", "14b"]):
        return """Guidelines:
- Be precise and factual
- Follow instructions carefully
- Cite sources when referencing information
- Be concise yet complete

"""

    return ""


def enhance_agent_system_prompt(
    user_system_prompt: str,
    model_name: Optional[str] = None,
) -> str:
    """
    Enhance user's custom agent system prompt with model-specific base instructions.

    This preserves the user's intent while adding foundational behavioral hints
    for small models. The base instructions are prepended so they act as a
    "grounding layer" without overriding user's actual prompt.

    **Usage**: Agent mode, workflows, tasks - where users write custom prompts
    **Don't use for**: RAG queries (use get_system_prompt_for_model instead)

    Args:
        user_system_prompt: User's custom system prompt
        model_name: Name of the model being used

    Returns:
        Enhanced system prompt with base instructions prepended

    Example:
        user_prompt = "You are a research assistant specializing in medical papers."
        model = "llama3.2:1b"

        enhanced = enhance_agent_system_prompt(user_prompt, model)
        # Result:
        # "Core Behavioral Rules:
        # - Follow instructions precisely as given
        # - Never make up information or guess
        # ...
        #
        # You are a research assistant specializing in medical papers."
    """
    if not user_system_prompt or not model_name:
        return user_system_prompt

    base_instructions = get_model_base_instructions(model_name)

    if not base_instructions:
        # Large model or no special instructions needed
        return user_system_prompt

    # Prepend base instructions to user's prompt
    # Add separator for clarity
    return f"{base_instructions}\n{user_system_prompt}"


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
