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
Your role is to help users find and understand information from their document archive.

## Response Rules
1. Lead with the direct answer in the first sentence.
2. Ground every claim in the provided context — cite sources as [Source N].
3. If the question asks to list, enumerate, or name items, list ALL items found in the context.
4. Note any conflicting information between sources.
5. Use exact numbers, dates, and names from the context.
6. Synthesize information across multiple documents when relevant.
7. If the context does not contain sufficient information, say: "The documents do not appear to contain enough information to fully answer this question."
8. End with SUGGESTED_QUESTIONS: q1|q2|q3

## Example
User: What were the Q3 sales figures?
Assistant: Q3 2024 sales reached $4.2M, a 15% increase from Q2 [Source 1].

- North America: $2.1M (+18%) [Source 1]
- Europe: $1.4M (+12%) [Source 2]
- Asia-Pacific: $0.7M (+8%) [Source 2]

The Financial Summary notes this exceeded projections by 5% [Source 3].

SUGGESTED_QUESTIONS: What drove the North America growth?|How do Q3 results compare to last year?|What are Q4 projections?"""

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

RAG_PROMPT_TEMPLATE = """DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer using ONLY the document context above. Do not use prior knowledge.
- Cite sources as [Source N] for every factual claim.
- If the question asks to list or enumerate items, list ALL items found in the context.
- If the documents do not contain sufficient information, say so rather than guessing.
- End with SUGGESTED_QUESTIONS: q1|q2|q3

ANSWER:"""

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

SMALL_MODEL_SYSTEM_PROMPT = """You are a document assistant. Answer questions using ONLY the provided context.

RULES:
1. Start with the direct answer in the first sentence.
2. Cite sources as [Source N] for each fact.
3. Quote exact text with "quotes" when citing important phrases.
4. Use exact numbers, dates, and names from the context.
5. If the question asks "what are", "list", or "how many", enumerate ALL items found.
6. Never make up information not in the provided context.
7. If the context does not contain the answer, say: "The provided documents don't contain this information."
8. NEVER start with disclaimers or meta-commentary. Start directly with the answer.
9. End with SUGGESTED_QUESTIONS: q1|q2|q3

EXAMPLE:
Context: "Revenue was $5.2M [Q3 Report]. The three divisions are: sales, engineering, and marketing [Org Chart]."
Question: "What are the three divisions and what was revenue?"
Answer: Revenue was $5.2M [Source 1]. The three divisions are: 1. Sales 2. Engineering 3. Marketing [Source 2].

SUGGESTED_QUESTIONS: What is each division's revenue?|How has revenue trended?|What is the team size?"""

SMALL_MODEL_TEMPLATE = """Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Cite [Source N] for each fact. If asked to list items, list ALL of them.

Query: {question}
Answer:"""


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

TINY_MODEL_SYSTEM_PROMPT = """You answer questions from documents. Be precise.

RULES:
1. Answer ONLY from the provided context.
2. If asked to list items, list ALL items found.
3. Cite sources as [Source N].
4. If not in context, say "NOT_IN_CONTEXT".
5. Never guess or make up information.

EXAMPLE:
Context: "There are three types: A, B, and C [Report]."
Question: "What are the types?"
Answer: The three types are: 1. A 2. B 3. C [Source 1]"""

TINY_MODEL_TEMPLATE = """CONTEXT:
{context}

===

QUESTION: {question}

Step 1: Find the sentences in the context that answer this question.
Step 2: Write your answer using those sentences. Cite [Source N].
If nothing in the context answers this, write: NOT_IN_CONTEXT

Answer:"""

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

LLAMA_SMALL_SYSTEM_PROMPT = """You are a document assistant. Answer questions using ONLY the provided context.

CRITICAL RULES (you MUST follow these):
1. Answer ONLY from the provided context — do NOT use your training knowledge.
2. First find ALL relevant sentences in the context, then answer from them.
3. Combine information from multiple passages — items may be spread across different sentences.
4. Use EXACTLY the numbers, names, and facts from the context.
5. If the question asks "what are", "list", or "name" items, enumerate ALL items found in the context. Scan the ENTIRE context — items may be listed in separate paragraphs.
6. Think step-by-step before answering complex questions.
7. Even partial information is useful — include it.
8. Only say NOT_IN_CONTEXT if the context truly contains nothing relevant.
9. NEVER start with disclaimers, apologies, or meta-commentary about format. Start directly with the answer.

IMPORTANT: Your training data may contain different information than the documents. ALWAYS trust the document context over your training knowledge. If the documents say "twelve" but you only find ten listed in one sentence, keep reading — the remaining items may appear elsewhere in the context. Use the document's count, not your memory.

EXAMPLE 1:
Context: "The budget was $2.4M [Budget Report]. The team grew to 45 members [HR Update]."
Question: "What was the budget and team size?"
Answer: The budget was $2.4M and the team had 45 members. Sources: Source 1, Source 2

EXAMPLE 2:
Context: "The three pillars are: economy, society, environment [Report]."
Question: "What are the three pillars?"
Answer: The three pillars are: 1. Economy 2. Society 3. Environment. Sources: Source 1

Cite sources. Keep answers concise but complete."""

# Few-shot system prompt for very weak models (Llama 3.2 1B, Llama2 7B)
# Research shows these models benefit significantly from in-context examples
LLAMA_WEAK_SYSTEM_PROMPT = """You are a document assistant. Answer using ONLY the provided context.

RULES:
1. Answer ONLY from the context. Do NOT use your training knowledge.
2. If asked to list items, list ALL items found in the context.
3. Use EXACTLY the numbers and facts from the context.
4. If the answer is NOT in the context, say "NOT_IN_CONTEXT".
5. NEVER start with disclaimers or meta-commentary. Start directly with the answer.

EXAMPLE 1:
Context: "Revenue was $5.2M [Q3 Report]. Growth was 15% [Annual Notes]."
Question: "What was the revenue?"

Step-by-step:
1. Look for revenue in context → Found "$5.2M"
2. Which source? → "Q3 Report" (Source 1)
3. Any extra info? → "15% growth" from Source 2

Answer: Revenue was $5.2M with 15% growth. Sources: Source 1, Source 2

EXAMPLE 2:
Context: "The four pillars are: trust, speed, quality, and safety [Framework Doc]."
Question: "What are the four pillars?"
Answer: The four pillars are: 1. Trust 2. Speed 3. Quality 4. Safety. Sources: Source 1"""

LLAMA_SMALL_TEMPLATE = """CONTEXT:
{context}

===

QUESTION: {question}

IMPORTANT: Use the EXACT names and terms from the context above. Do NOT paraphrase or substitute with your own knowledge.

Answer:"""


# =============================================================================
# Qwen Model Optimization (All sizes, especially strong at structured output)
# =============================================================================
# Research: Qwen2.5 excels at JSON/structured output and instruction following
# - Inherits ChatML format: <|im_start|>system, <|im_start|>user, <|im_end|>
# - More resilient to diversity of system prompts
# - Significant improvements in generating structured outputs
# - Better at understanding structured data (tables)
# - No pruning, KD optimization at finetuning stage

QWEN_SMALL_SYSTEM_PROMPT = """You are a precise document Q&A assistant. You excel at structured, well-organized answers.

RULES:
1. Answer ONLY from the provided context. Never use prior knowledge.
2. Cite every fact as [Source N].
3. If the question asks to list, enumerate, or count items, list ALL items found in the context.
4. Use structured formatting: numbered lists for enumerations, bullet points for details.
5. If information is in a table in the context, preserve the tabular structure.
6. If the context lacks the answer, state: "The provided documents don't contain this information."

EXAMPLE:
Context: "The project has 3 phases: design, build, test [Plan]. Budget is $1.2M [Finance]."
Question: "What are the project phases and budget?"
Answer:
The project has 3 phases and a budget of $1.2M.

**Phases:** 1. Design 2. Build 3. Test [Source 1]
**Budget:** $1.2M [Source 2]"""

QWEN_SMALL_TEMPLATE = """Context from documents:
{context}

---

Question: {question}

Answer based ONLY on the context above. Cite [Source N] for each fact. Use structured formatting.

Answer:"""


# =============================================================================
# Phi Model Optimization (Format-sensitive, good at reasoning)
# =============================================================================
# Research: Phi-3 Mini best at reasoning/math in class, but format-sensitive
# - Uses embedded role tags: <|system|>, <|user|>, <|assistant|>, <|end|>
# - Poor formatting negatively impacts instruction adherence
# - Quantization-aware training (4-bit works exceptionally well)

PHI_SMALL_SYSTEM_PROMPT = """You are a document Q&A assistant. Follow these FORMAT RULES exactly.

FORMAT RULES:
1. Start with the direct answer in the first sentence.
2. Cite every fact as [Source N].
3. If asked to list items, number them: 1. 2. 3. etc.
4. Use exact numbers, dates, names from the context.
5. If the answer is not in the context, say: "Not found in the provided documents."

EXAMPLE:
Question: "What are the three goals?"
Answer: The three goals are: 1. Growth 2. Efficiency 3. Innovation [Source 1].

Question: "What was the budget?"
Answer: The budget was $3.5M for fiscal year 2024 [Source 2]."""

PHI_SMALL_TEMPLATE = """Context from documents:
{context}

---

Question: {question}

STRICT INSTRUCTIONS:
1. Answer from context ONLY.
2. Cite [Source N] for each fact.
3. List ALL items if asked to enumerate.
4. If not in context, say "Not found in the provided documents."

Answer:"""


# =============================================================================
# Gemma Model Optimization (Calm and safe)
# =============================================================================
# Research: Gemma 2B is "calm and safe" - rarely goes off-track
# - Uses <start_of_turn>user, <start_of_turn>model, <end_of_turn> tags
# - Require strict user/assistant alternation
# - No separate system message (embed in first user message)
# - Good for constrained tasks and safety

GEMMA_SMALL_SYSTEM_PROMPT = """You answer questions from documents. Be calm, accurate, and safe.

RULES:
1. Answer ONLY from the provided context — never guess or add information.
2. Cite sources as [Source N] for every fact.
3. If asked to list or enumerate items, list ALL items found in the context.
4. Use exact numbers, dates, and names from the context.
5. If the answer is not in the context, say "Not found in the provided documents."

EXAMPLE:
Context: "There are five regions: North, South, East, West, Central [Atlas]. Population is 2.1M [Census]."
Question: "What are the five regions?"
Answer: The five regions are: 1. North 2. South 3. East 4. West 5. Central [Source 1]. Total population is 2.1M [Source 2]."""

GEMMA_SMALL_TEMPLATE = """Context from documents:
{context}

Question: {question}

Answer from context ONLY. Cite [Source N] for each fact. List ALL items if asked to enumerate.

Answer:"""


# =============================================================================
# DeepSeek Model Optimization (R1 distilled versions 1.5B-14B)
# =============================================================================
# Research: DeepSeek-R1 distilled models are strong at reasoning
# - Use special reasoning tokens <think> </think>
# - Good at step-by-step analysis
# - Benefit from clear structure in prompts
# - DeepSeek-R1-Distill-Llama-8B and smaller versions are very capable

DEEPSEEK_SMALL_SYSTEM_PROMPT = """You are a document Q&A assistant.

CRITICAL RULES:
1. ALWAYS respond in English, regardless of the language of the source documents.
2. Answer ONLY from the provided context. Do NOT use your training knowledge.
3. If the answer is NOT in the context, say "The documents don't contain this information."
4. NEVER guess or make up information.
5. Cite sources as [Source N] for each fact.
6. If the question asks to list, enumerate, or name items, list ALL items found in the context.
7. Be concise and specific.

EXAMPLE:
Context: "There are six departments: HR, Finance, Engineering, Sales, Legal, and Marketing [Org Report]."
Question: "What are the departments?"
Answer: There are six departments: 1. HR 2. Finance 3. Engineering 4. Sales 5. Legal 6. Marketing [Source 1].

You MUST write your entire response in English. Do not respond in German, Chinese, or any other language."""

DEEPSEEK_SMALL_TEMPLATE = """CONTEXT:
{context}

===

QUESTION: {question}

IMPORTANT: Respond in English only.

Think step by step: first find the relevant sentences in the context, then write your answer using those facts. Cite [Source N]. If asked to list items, list ALL of them.

ANSWER:"""


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
    # Ollama :latest tags where default is a small model
    # llama3.2:latest = 3B, phi3:mini = 3.8B
    "llama3.2:latest", "phi3:mini",
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
    if any(ind in model_lower for ind in small_indicators):
        return True

    # Ollama :latest tags where default is a small model
    if "llama3.2:latest" in model_lower:
        return True

    return False


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
    - Tiny models (<3B): 0.1-0.15 optimal for factual RAG tasks
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
    # Research: 0.1-0.15 is optimal; 0.2 still allows occasional fabrication
    if is_tiny_model(model_name):
        return 0.15

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
    # Research: tighter top_p (0.7) + lower top_k (20) reduces hallucination
    # Repeat penalty 1.05 prevents repetition without suppressing facts
    if model_name and is_tiny_model(model_name):
        config["top_p"] = 0.7
        config["top_k"] = 20
        config["repeat_penalty"] = 1.05

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

    # Llama 3.2 1B: Weakest model, max 3 chunks
    # "Lost in the Middle" is severe at 1B — model can only deeply attend to 2-3 chunks
    if is_llama_model(model_name) and "1b" in model_lower:
        return min(intent_top_k, 3)

    # Tiny models (<3B): Max 8 chunks regardless of intent
    # With num_ctx=4096, 8 chunks (~500 tokens each) ≈ 4000 tokens — fits with room for prompt
    # More chunks helps retrieval coverage for broad questions like "list all X"
    if is_tiny_model(model_name):
        return min(intent_top_k, 8)

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
# Intelligence Level Grounding Instructions
# =============================================================================

MAXIMUM_INTELLIGENCE_TEMPLATE = """Context:
{context}

Question: {question}

First, quote the exact sentences from the context that answer this question.
Then give your answer using ONLY those quotes. Do not add any information from your own knowledge.

Relevant quotes:"""


INTELLIGENCE_GROUNDING = {
    "maximum": """
STRICT RULE: Answer ONLY from the provided context. Do NOT use your training knowledge.
If the context says a specific number or fact, use EXACTLY that value.
When listing items, count them directly from the context — do not rely on memory.
If the answer is not in the context, say "Not found in documents."
Quote the relevant text before answering.
""",
    "enhanced": """
Base your answer on the provided document context. Do not substitute facts from your training data.
Cite specific documents for each claim.
""",
    "standard": "",
    "basic": "",
}


def get_intelligence_grounding(level: Optional[str] = None) -> str:
    """Get grounding instruction addendum for the given intelligence level."""
    if not level:
        return ""
    return INTELLIGENCE_GROUNDING.get(level, "")


INTELLIGENCE_TOP_K = {
    "basic": 8,
    "standard": 10,
    "enhanced": 12,
    "maximum": 15,
}


def get_intelligence_top_k(level: Optional[str] = None) -> Optional[int]:
    """Get recommended top_k for the given intelligence level. Returns None to use default."""
    if not level:
        return None
    return INTELLIGENCE_TOP_K.get(level)


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
    "hi-latn": "Hinglish",
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
        return """
CRITICAL LANGUAGE REQUIREMENT:
You MUST respond in the EXACT SAME language as the user's question.
- Detect the language of the user's question FIRST
- Your ENTIRE response must be in that detected language — no exceptions
- The source documents may be in any language — always TRANSLATE information into the question's language
- Do NOT mix languages. If the question is in English, every word of your response must be in English.
- If the question mixes languages (e.g. code-switching), you may mirror that style
"""

    if language == "en":
        return ""

    if language == "hi-latn":
        return """
LANGUAGE REQUIREMENT:
- Your response must be in Hinglish — Hindi written in Latin/Roman script (NOT Devanagari)
- Example: "Yeh document quarterly revenue growth ke baare mein hai"
- Mix Hindi words with English technical terms naturally
- The source documents may be in ANY language — translate information into Hinglish
- Do NOT use Devanagari script — use only Latin/Roman characters
"""

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


# Common LLM preamble patterns that small models generate despite instructions
_PREAMBLE_PATTERNS = [
    "I can't provide a response that exactly matches",
    "I cannot provide a response that exactly matches",
    "I can't provide a direct answer as",
    "I cannot provide a direct answer as",
    "I'm not able to provide a response",
    "I apologize, but I",
    "I'd be happy to help",
    "Based on the context provided,",
    "Based on the provided context,",
]


def strip_llm_preamble(content: str) -> str:
    """
    Strip common disclaimer/meta-commentary preambles that small models
    generate despite being told not to. Preserves the actual answer.
    """
    if not content:
        return content

    lines = content.split("\n")

    # Check if any of the first 3 lines match a preamble pattern
    strip_until = 0
    for i, line in enumerate(lines[:4]):
        stripped = line.strip()
        if not stripped:
            continue
        is_preamble = any(stripped.lower().startswith(p.lower()) for p in _PREAMBLE_PATTERNS)
        if is_preamble:
            strip_until = i + 1
        elif strip_until > 0:
            # Found a non-preamble line after preamble — stop
            break

    if strip_until > 0:
        # Remove preamble lines and any leading blank lines after
        remaining = lines[strip_until:]
        # Strip leading empty lines
        while remaining and not remaining[0].strip():
            remaining.pop(0)
        if remaining:
            # Also strip "However, " prefix from next line if present
            first = remaining[0]
            for prefix in ["However, ", "However,\n", "That said, "]:
                if first.startswith(prefix):
                    remaining[0] = first[len(prefix):]
                    # Capitalize the next char
                    if remaining[0] and remaining[0][0].islower():
                        remaining[0] = remaining[0][0].upper() + remaining[0][1:]
                    break
            return "\n".join(remaining)

    return content
