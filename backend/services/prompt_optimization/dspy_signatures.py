"""
AIDocumentIndexer - DSPy Signature & Module Definitions
=======================================================

Phase 93: DSPy prompt optimization integration.

Defines DSPy Signatures (input/output specs) and Modules (composable prompt
components) that mirror the system's existing RAG prompts. These are used by
the DSPy optimizer to discover optimal instructions and few-shot examples.

Signatures:
1. RAGAnswerGeneration - Core RAG answer with citations
2. QueryExpansion - Search query variation generation
3. QueryDecomposition - Complex query decomposition
4. ReActReasoning - Agentic RAG reasoning steps
5. AnswerSynthesis - Multi-source answer synthesis
"""

import structlog

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = structlog.get_logger(__name__)


# =============================================================================
# DSPy Signatures
# =============================================================================

if DSPY_AVAILABLE:

    class RAGAnswerGeneration(dspy.Signature):
        """Answer questions based on retrieved document context.
        Cite sources with [Document Name]. Be specific with numbers, dates, names.
        If context lacks information, say so clearly."""

        context: str = dspy.InputField(
            desc="Retrieved document chunks with source attribution"
        )
        question: str = dspy.InputField(
            desc="User's question about documents"
        )
        answer: str = dspy.OutputField(
            desc="Accurate answer citing [Document Name] sources"
        )
        suggested_questions: str = dspy.OutputField(
            desc="2-3 follow-up questions separated by |"
        )

    class QueryExpansion(dspy.Signature):
        """Generate search query variations to improve document retrieval accuracy.
        Produce diverse reformulations using synonyms, related concepts, and
        different phrasings while preserving the original intent."""

        original_query: str = dspy.InputField(
            desc="User's original search query"
        )
        num_variations: int = dspy.InputField(
            desc="Number of query variations to generate"
        )
        expanded_queries: str = dspy.OutputField(
            desc="JSON list of query variations"
        )

    class QueryDecomposition(dspy.Signature):
        """Decompose complex queries into atomic sub-questions for multi-step retrieval.
        Identify if decomposition is needed and create focused sub-queries with
        dependency tracking."""

        query: str = dspy.InputField(desc="Complex user query")
        is_complex: bool = dspy.OutputField(
            desc="Whether decomposition is needed"
        )
        sub_queries: str = dspy.OutputField(
            desc="JSON list of sub-queries with purposes and dependencies"
        )
        synthesis_approach: str = dspy.OutputField(
            desc="How to combine sub-answers into a final response"
        )

    class ReActReasoning(dspy.Signature):
        """Reason about the next action in document retrieval using the ReAct framework.
        Choose the most effective action based on what has been learned so far."""

        query: str = dspy.InputField(desc="Current question being investigated")
        previous_steps: str = dspy.InputField(desc="Steps taken so far")
        current_knowledge: str = dspy.InputField(
            desc="Information gathered so far"
        )
        thought: str = dspy.OutputField(
            desc="Reasoning about what to do next"
        )
        action: str = dspy.OutputField(
            desc="Action: search, graph, summarize, compare, or answer"
        )
        action_input: str = dspy.OutputField(
            desc="Input for the chosen action"
        )

    class AnswerSynthesis(dspy.Signature):
        """Synthesize information from multiple retrieval steps into a comprehensive answer.
        Integrate sub-answers, graph context, and raw document chunks into a coherent
        response with proper citations."""

        query: str = dspy.InputField(desc="Original user question")
        sub_answers: str = dspy.InputField(
            desc="Answers to decomposed sub-questions"
        )
        graph_context: str = dspy.InputField(
            desc="Knowledge graph entity and relation context"
        )
        retrieved_context: str = dspy.InputField(
            desc="Raw retrieved document chunks"
        )
        answer: str = dspy.OutputField(
            desc="Comprehensive synthesized answer with citations"
        )

    # =========================================================================
    # DSPy Modules (Composable Prompt Components)
    # =========================================================================

    class RAGAnswerModule(dspy.Module):
        """RAG answer generation with chain-of-thought reasoning."""

        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(RAGAnswerGeneration)

        def forward(self, context: str, question: str):
            return self.generate(context=context, question=question)

    class QueryExpansionModule(dspy.Module):
        """Query expansion using direct prediction."""

        def __init__(self):
            super().__init__()
            self.expand = dspy.Predict(QueryExpansion)

        def forward(self, original_query: str, num_variations: int = 3):
            return self.expand(
                original_query=original_query,
                num_variations=num_variations,
            )

    class QueryDecompositionModule(dspy.Module):
        """Query decomposition with chain-of-thought."""

        def __init__(self):
            super().__init__()
            self.decompose = dspy.ChainOfThought(QueryDecomposition)

        def forward(self, query: str):
            return self.decompose(query=query)

    class ReActReasoningModule(dspy.Module):
        """ReAct reasoning step."""

        def __init__(self):
            super().__init__()
            self.reason = dspy.ChainOfThought(ReActReasoning)

        def forward(
            self,
            query: str,
            previous_steps: str = "",
            current_knowledge: str = "",
        ):
            return self.reason(
                query=query,
                previous_steps=previous_steps,
                current_knowledge=current_knowledge,
            )

    class AnswerSynthesisModule(dspy.Module):
        """Multi-source answer synthesis."""

        def __init__(self):
            super().__init__()
            self.synthesize = dspy.ChainOfThought(AnswerSynthesis)

        def forward(
            self,
            query: str,
            sub_answers: str = "",
            graph_context: str = "",
            retrieved_context: str = "",
        ):
            return self.synthesize(
                query=query,
                sub_answers=sub_answers,
                graph_context=graph_context,
                retrieved_context=retrieved_context,
            )

    class AgenticRAGModule(dspy.Module):
        """
        Full agentic RAG pipeline composing decomposition, reasoning, and synthesis.
        Used for end-to-end optimization of the agentic RAG pipeline.
        """

        def __init__(self):
            super().__init__()
            self.decompose = QueryDecompositionModule()
            self.reason = ReActReasoningModule()
            self.synthesize = AnswerSynthesisModule()

        def forward(self, query: str, context: str = ""):
            # Step 1: Decompose if complex
            decomp = self.decompose(query=query)

            # Step 2: Reason about next action
            reasoning = self.reason(
                query=query,
                previous_steps="Decomposition complete",
                current_knowledge=context,
            )

            # Step 3: Synthesize final answer
            return self.synthesize(
                query=query,
                sub_answers=str(decomp.sub_queries),
                graph_context="",
                retrieved_context=context,
            )


# =============================================================================
# Module Registry
# =============================================================================

SIGNATURE_REGISTRY = {
    "rag_answer": "RAGAnswerGeneration",
    "query_expansion": "QueryExpansion",
    "query_decomposition": "QueryDecomposition",
    "react_reasoning": "ReActReasoning",
    "answer_synthesis": "AnswerSynthesis",
}

MODULE_REGISTRY = {
    "rag_answer": "RAGAnswerModule",
    "query_expansion": "QueryExpansionModule",
    "query_decomposition": "QueryDecompositionModule",
    "react_reasoning": "ReActReasoningModule",
    "answer_synthesis": "AnswerSynthesisModule",
    "agentic_rag": "AgenticRAGModule",
}


def get_module(name: str):
    """Get a DSPy module by name."""
    if not DSPY_AVAILABLE:
        raise ImportError("dspy-ai is required for prompt optimization. Install with: pip install dspy-ai")

    module_name = MODULE_REGISTRY.get(name)
    if not module_name:
        raise ValueError(f"Unknown module: {name}. Available: {list(MODULE_REGISTRY.keys())}")

    return globals()[module_name]()
