"""
AIDocumentIndexer - Tree of Thoughts Service
==============================================

Implements Tree of Thoughts (ToT) and Best-of-N for complex reasoning.

Tree of Thoughts (ToT) - NeurIPS 2023:
- Explore multiple reasoning paths simultaneously
- Evaluate and prune unpromising branches
- 4% → 74% on Game of 24 (complex math)

Best-of-N with Reward Models:
- Generate N candidate responses
- Score each with reward model
- Return highest-scoring response
- Used in RLHF and constitutional AI

Research:
- "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (NeurIPS 2023)
- "Training Language Models with Language Feedback at Scale" (Anthropic)
- "Process Reward Models" (OpenAI)

Performance:
| Method | Game of 24 | Creative Writing | Mini Crossword |
|--------|------------|------------------|----------------|
| Standard | 4% | 6.2 | 16% |
| CoT | 4% | 6.9 | 16% |
| ToT | 74% | 7.6 | 60% |
"""

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from backend.core.config import settings
from backend.core.performance import gather_with_concurrency

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class SearchStrategy(str, Enum):
    """Search strategies for Tree of Thoughts."""
    BFS = "bfs"           # Breadth-first (explore all at each level)
    DFS = "dfs"           # Depth-first (explore one path deeply)
    BEAM = "beam"         # Beam search (keep top-k at each level)
    MCTS = "mcts"         # Monte Carlo Tree Search


@dataclass
class ToTConfig:
    """Configuration for Tree of Thoughts."""
    # Tree structure
    max_depth: int = 3              # Maximum tree depth
    branching_factor: int = 3       # Thoughts per node
    beam_width: int = 3             # Candidates kept at each level (for beam search)

    # Evaluation
    evaluation_method: str = "value"  # "value" or "vote"
    value_threshold: float = 0.5      # Prune nodes below this value

    # Search strategy
    search_strategy: SearchStrategy = SearchStrategy.BEAM

    # Model settings
    thought_model: str = "gpt-4o-mini"
    thought_provider: str = "openai"
    evaluation_model: str = "gpt-4o-mini"
    evaluation_provider: str = "openai"
    temperature: float = 0.7          # Higher for diverse thoughts

    # Best-of-N settings
    best_of_n: int = 3               # Generate N candidates for final answer


@dataclass
class BestOfNConfig:
    """Configuration for Best-of-N sampling."""
    n_samples: int = 5              # Number of candidates to generate
    temperature: float = 0.8        # Temperature for diversity
    model: str = "gpt-4o-mini"
    provider: str = "openai"

    # Reward model
    use_reward_model: bool = False
    reward_model: str = "gpt-4o-mini"  # Use LLM as reward model


@dataclass(slots=True)
class ThoughtNode:
    """A node in the thought tree."""
    id: str
    thought: str                    # The reasoning step
    state: str                      # Current problem state
    value: float = 0.0              # Evaluation score
    depth: int = 0
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    is_terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToTResult:
    """Result from Tree of Thoughts."""
    answer: str
    best_path: List[ThoughtNode]
    all_paths: List[List[ThoughtNode]]
    nodes_explored: int
    max_depth_reached: int
    execution_time_ms: float
    confidence: float


@dataclass(slots=True)
class BestOfNResult:
    """Result from Best-of-N sampling."""
    best_response: str
    all_responses: List[str]
    scores: List[float]
    selected_index: int
    execution_time_ms: float


# =============================================================================
# Prompts
# =============================================================================

THOUGHT_GENERATION_PROMPT = """You are solving a problem step by step.

Problem: {problem}

Current state/progress:
{state}

Generate {n} distinct next steps or thoughts that could help solve this problem.
Each thought should be a single reasoning step, not the final answer.

Format each thought on a new line starting with "Thought N:".

Thought 1:"""

THOUGHT_EVALUATION_PROMPT = """Evaluate this reasoning step for solving the problem.

Problem: {problem}

Current state: {state}

Proposed thought/step: {thought}

Rate this thought on a scale of 1-10:
- Is it logically sound?
- Does it make progress toward the solution?
- Is it likely to lead to the correct answer?

Score (1-10):"""

SOLUTION_SYNTHESIS_PROMPT = """Based on this reasoning path, provide the final answer.

Problem: {problem}

Reasoning steps:
{reasoning_path}

Final Answer:"""

REWARD_MODEL_PROMPT = """Rate this response for quality, accuracy, and helpfulness.

Query: {query}

Response: {response}

Consider:
1. Accuracy: Is the information correct?
2. Completeness: Does it fully address the query?
3. Clarity: Is it well-written and easy to understand?
4. Helpfulness: Would this be useful to the user?

Overall Score (1-10):"""


# =============================================================================
# Tree of Thoughts
# =============================================================================

class TreeOfThoughts:
    """
    Tree of Thoughts for complex reasoning.

    Explores multiple reasoning paths and evaluates each step,
    pruning unpromising branches early.

    Usage:
        tot = TreeOfThoughts()

        result = await tot.solve(
            problem="Find a solution where 4 numbers make 24 using +,-,*,/",
            initial_state="Numbers: 1, 2, 3, 4"
        )
    """

    def __init__(self, config: Optional[ToTConfig] = None):
        self.config = config or ToTConfig()
        self._thought_llm = None
        self._eval_llm = None
        self._nodes: Dict[str, ThoughtNode] = {}
        self._node_counter = 0
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize LLMs."""
        if self._initialized:
            return True

        try:
            from backend.services.llm import LLMFactory

            self._thought_llm = LLMFactory.get_chat_model(
                provider=self.config.thought_provider,
                model=self.config.thought_model,
                temperature=self.config.temperature,
                max_tokens=1024,
            )

            self._eval_llm = LLMFactory.get_chat_model(
                provider=self.config.evaluation_provider,
                model=self.config.evaluation_model,
                temperature=0.1,  # Low temperature for evaluation
                max_tokens=256,
            )

            logger.info(
                "Tree of Thoughts initialized",
                thought_model=self.config.thought_model,
                search_strategy=self.config.search_strategy.value,
            )

            self._initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize ToT", error=str(e))
            return False

    async def solve(
        self,
        problem: str,
        initial_state: str = "",
        context: Optional[str] = None,
    ) -> ToTResult:
        """
        Solve a problem using Tree of Thoughts.

        Args:
            problem: The problem to solve
            initial_state: Initial problem state
            context: Optional context/background

        Returns:
            ToTResult with best solution path
        """
        if not await self.initialize():
            return ToTResult(
                answer="",
                best_path=[],
                all_paths=[],
                nodes_explored=0,
                max_depth_reached=0,
                execution_time_ms=0,
                confidence=0,
            )

        start_time = time.time()
        self._nodes = {}
        self._node_counter = 0

        # Create root node
        root = self._create_node(
            thought="Start",
            state=initial_state or problem,
            depth=0,
        )

        logger.info(
            "Starting Tree of Thoughts",
            problem_length=len(problem),
            strategy=self.config.search_strategy.value,
        )

        # Run search based on strategy
        if self.config.search_strategy == SearchStrategy.BFS:
            best_path = await self._bfs_search(problem, root)
        elif self.config.search_strategy == SearchStrategy.DFS:
            best_path = await self._dfs_search(problem, root)
        elif self.config.search_strategy == SearchStrategy.BEAM:
            best_path = await self._beam_search(problem, root)
        else:
            best_path = await self._beam_search(problem, root)  # Default to beam

        # Synthesize final answer from best path
        answer = await self._synthesize_answer(problem, best_path)

        # Calculate confidence from path values
        confidence = sum(n.value for n in best_path) / len(best_path) if best_path else 0

        execution_time = (time.time() - start_time) * 1000

        logger.info(
            "Tree of Thoughts complete",
            nodes_explored=len(self._nodes),
            best_path_length=len(best_path),
            confidence=round(confidence, 2),
            time_ms=round(execution_time, 2),
        )

        return ToTResult(
            answer=answer,
            best_path=best_path,
            all_paths=[best_path],  # Could track all complete paths
            nodes_explored=len(self._nodes),
            max_depth_reached=max(n.depth for n in self._nodes.values()) if self._nodes else 0,
            execution_time_ms=execution_time,
            confidence=confidence / 10,  # Normalize to 0-1
        )

    async def _beam_search(
        self,
        problem: str,
        root: ThoughtNode,
    ) -> List[ThoughtNode]:
        """Beam search: keep top-k candidates at each level."""
        beam = [root]

        for depth in range(self.config.max_depth):
            if not beam:
                break

            # Generate thoughts for all nodes in beam
            all_candidates = []

            for node in beam:
                thoughts = await self._generate_thoughts(problem, node)
                for thought in thoughts:
                    child = self._create_node(
                        thought=thought,
                        state=f"{node.state}\n-> {thought}",
                        depth=depth + 1,
                        parent_id=node.id,
                    )
                    node.children.append(child.id)
                    all_candidates.append(child)

            if not all_candidates:
                break

            # Evaluate all candidates
            await self._evaluate_nodes(problem, all_candidates)

            # Keep top-k by value
            all_candidates.sort(key=lambda n: n.value, reverse=True)
            beam = all_candidates[:self.config.beam_width]

            # Check for terminal nodes
            for node in beam:
                if await self._is_terminal(problem, node):
                    node.is_terminal = True

            # If any terminal, stop
            terminal = [n for n in beam if n.is_terminal]
            if terminal:
                beam = terminal
                break

        # Return best path
        if beam:
            best_node = max(beam, key=lambda n: n.value)
            return self._reconstruct_path(best_node)
        return [root]

    async def _bfs_search(
        self,
        problem: str,
        root: ThoughtNode,
    ) -> List[ThoughtNode]:
        """Breadth-first search: explore all nodes at each level."""
        queue = [root]
        best_terminal = None
        best_value = float('-inf')

        while queue:
            node = queue.pop(0)

            if node.depth >= self.config.max_depth:
                if node.value > best_value:
                    best_value = node.value
                    best_terminal = node
                continue

            # Generate and evaluate children
            thoughts = await self._generate_thoughts(problem, node)

            for thought in thoughts:
                child = self._create_node(
                    thought=thought,
                    state=f"{node.state}\n-> {thought}",
                    depth=node.depth + 1,
                    parent_id=node.id,
                )
                node.children.append(child.id)

                # Evaluate
                child.value = await self._evaluate_thought(problem, child)

                # Prune low-value nodes
                if child.value < self.config.value_threshold * 10:
                    continue

                if await self._is_terminal(problem, child):
                    child.is_terminal = True
                    if child.value > best_value:
                        best_value = child.value
                        best_terminal = child
                else:
                    queue.append(child)

        if best_terminal:
            return self._reconstruct_path(best_terminal)
        return [root]

    async def _dfs_search(
        self,
        problem: str,
        root: ThoughtNode,
    ) -> List[ThoughtNode]:
        """Depth-first search with pruning."""
        best_path = [root]
        best_value = float('-inf')

        async def dfs(node: ThoughtNode, path: List[ThoughtNode]):
            nonlocal best_path, best_value

            if node.depth >= self.config.max_depth:
                path_value = sum(n.value for n in path)
                if path_value > best_value:
                    best_value = path_value
                    best_path = path.copy()
                return

            thoughts = await self._generate_thoughts(problem, node)

            for thought in thoughts:
                child = self._create_node(
                    thought=thought,
                    state=f"{node.state}\n-> {thought}",
                    depth=node.depth + 1,
                    parent_id=node.id,
                )
                node.children.append(child.id)
                child.value = await self._evaluate_thought(problem, child)

                # Prune
                if child.value < self.config.value_threshold * 10:
                    continue

                path.append(child)

                if await self._is_terminal(problem, child):
                    path_value = sum(n.value for n in path)
                    if path_value > best_value:
                        best_value = path_value
                        best_path = path.copy()
                else:
                    await dfs(child, path)

                path.pop()

        await dfs(root, [root])
        return best_path

    async def _generate_thoughts(
        self,
        problem: str,
        node: ThoughtNode,
    ) -> List[str]:
        """Generate multiple next thoughts from a node."""
        from langchain_core.messages import HumanMessage

        prompt = THOUGHT_GENERATION_PROMPT.format(
            problem=problem,
            state=node.state,
            n=self.config.branching_factor,
        )

        try:
            response = await self._thought_llm.ainvoke([HumanMessage(content=prompt)])
            thoughts = self._parse_thoughts(response.content)
            return thoughts[:self.config.branching_factor]
        except Exception as e:
            logger.warning("Thought generation failed", error=str(e))
            return []

    def _parse_thoughts(self, text: str) -> List[str]:
        """Parse thoughts from LLM response."""
        thoughts = []
        lines = text.strip().split('\n')

        current_thought = []
        for line in lines:
            if line.strip().startswith("Thought"):
                if current_thought:
                    thoughts.append(' '.join(current_thought).strip())
                # Extract thought after "Thought N:"
                parts = line.split(':', 1)
                if len(parts) > 1:
                    current_thought = [parts[1].strip()]
                else:
                    current_thought = []
            elif current_thought is not None and line.strip():
                current_thought.append(line.strip())

        if current_thought:
            thoughts.append(' '.join(current_thought).strip())

        return [t for t in thoughts if t]

    async def _evaluate_thought(
        self,
        problem: str,
        node: ThoughtNode,
    ) -> float:
        """Evaluate a single thought node."""
        from langchain_core.messages import HumanMessage
        import re

        prompt = THOUGHT_EVALUATION_PROMPT.format(
            problem=problem,
            state=node.state[:500],
            thought=node.thought,
        )

        try:
            response = await self._eval_llm.ainvoke([HumanMessage(content=prompt)])
            # Extract score
            match = re.search(r'(\d+(?:\.\d+)?)', response.content)
            if match:
                return float(match.group(1))
            return 5.0  # Default
        except Exception:
            return 5.0

    async def _evaluate_nodes(
        self,
        problem: str,
        nodes: List[ThoughtNode],
    ) -> None:
        """Evaluate multiple nodes in parallel."""
        async def eval_one(node):
            node.value = await self._evaluate_thought(problem, node)

        await gather_with_concurrency(
            [eval_one(n) for n in nodes],
            max_concurrent=5,
        )

    async def _is_terminal(self, problem: str, node: ThoughtNode) -> bool:
        """Check if node represents a terminal/solution state."""
        # Simple heuristic: check for answer indicators
        indicators = ["answer is", "solution is", "therefore", "thus", "finally"]
        thought_lower = node.thought.lower()
        return any(ind in thought_lower for ind in indicators)

    async def _synthesize_answer(
        self,
        problem: str,
        path: List[ThoughtNode],
    ) -> str:
        """Synthesize final answer from reasoning path."""
        from langchain_core.messages import HumanMessage

        reasoning_path = "\n".join(
            f"{i+1}. {node.thought}"
            for i, node in enumerate(path)
            if node.thought != "Start"
        )

        prompt = SOLUTION_SYNTHESIS_PROMPT.format(
            problem=problem,
            reasoning_path=reasoning_path or "No reasoning steps recorded.",
        )

        try:
            response = await self._thought_llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception:
            # Fall back to last thought
            return path[-1].thought if path else ""

    def _create_node(
        self,
        thought: str,
        state: str,
        depth: int,
        parent_id: Optional[str] = None,
    ) -> ThoughtNode:
        """Create a new thought node."""
        self._node_counter += 1
        node_id = f"node_{self._node_counter}"

        node = ThoughtNode(
            id=node_id,
            thought=thought,
            state=state,
            depth=depth,
            parent_id=parent_id,
        )

        self._nodes[node_id] = node
        return node

    def _reconstruct_path(self, node: ThoughtNode) -> List[ThoughtNode]:
        """Reconstruct path from root to node."""
        path = []
        current = node

        while current:
            path.append(current)
            if current.parent_id:
                current = self._nodes.get(current.parent_id)
            else:
                current = None

        path.reverse()
        return path


# =============================================================================
# Best-of-N Sampling
# =============================================================================

class BestOfN:
    """
    Best-of-N sampling with reward model scoring.

    Generate N candidate responses and select the best one
    using a reward model (or LLM-as-judge).

    Usage:
        bon = BestOfN()

        result = await bon.generate(
            query="Explain quantum computing",
            context="...",
        )
    """

    def __init__(self, config: Optional[BestOfNConfig] = None):
        self.config = config or BestOfNConfig()
        self._generator_llm = None
        self._reward_llm = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize LLMs."""
        if self._initialized:
            return True

        try:
            from backend.services.llm import LLMFactory

            self._generator_llm = LLMFactory.get_chat_model(
                provider=self.config.provider,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=2048,
            )

            if self.config.use_reward_model:
                self._reward_llm = LLMFactory.get_chat_model(
                    provider=self.config.provider,
                    model=self.config.reward_model,
                    temperature=0.1,
                    max_tokens=256,
                )

            logger.info(
                "Best-of-N initialized",
                n_samples=self.config.n_samples,
                model=self.config.model,
            )

            self._initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize Best-of-N", error=str(e))
            return False

    async def generate(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> BestOfNResult:
        """
        Generate N responses and return the best one.

        Args:
            query: User query
            context: Optional context
            system_prompt: Optional system prompt

        Returns:
            BestOfNResult with best response
        """
        if not await self.initialize():
            return BestOfNResult(
                best_response="",
                all_responses=[],
                scores=[],
                selected_index=0,
                execution_time_ms=0,
            )

        start_time = time.time()

        logger.info(
            "Starting Best-of-N generation",
            n_samples=self.config.n_samples,
        )

        # Generate N responses in parallel
        responses = await self._generate_candidates(
            query, context, system_prompt
        )

        if not responses:
            return BestOfNResult(
                best_response="",
                all_responses=[],
                scores=[],
                selected_index=0,
                execution_time_ms=0,
            )

        # Score responses
        if self.config.use_reward_model:
            scores = await self._score_responses(query, responses)
        else:
            # Simple length-based scoring as fallback
            scores = [len(r) / 1000 for r in responses]

        # Select best
        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        execution_time = (time.time() - start_time) * 1000

        logger.info(
            "Best-of-N complete",
            n_generated=len(responses),
            best_score=round(scores[best_idx], 2),
            time_ms=round(execution_time, 2),
        )

        return BestOfNResult(
            best_response=responses[best_idx],
            all_responses=responses,
            scores=scores,
            selected_index=best_idx,
            execution_time_ms=execution_time,
        )

    async def _generate_candidates(
        self,
        query: str,
        context: Optional[str],
        system_prompt: Optional[str],
    ) -> List[str]:
        """Generate N candidate responses."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        user_content = query
        if context:
            user_content = f"Context:\n{context}\n\nQuery: {query}"

        messages.append(HumanMessage(content=user_content))

        # Generate in parallel
        async def generate_one():
            response = await self._generator_llm.ainvoke(messages)
            return response.content

        tasks = [generate_one() for _ in range(self.config.n_samples)]
        results = await gather_with_concurrency(tasks, max_concurrent=5)

        return [r for r in results if isinstance(r, str)]

    async def _score_responses(
        self,
        query: str,
        responses: List[str],
    ) -> List[float]:
        """Score responses using reward model."""
        from langchain_core.messages import HumanMessage
        import re

        async def score_one(response: str) -> float:
            prompt = REWARD_MODEL_PROMPT.format(
                query=query,
                response=response[:2000],
            )

            try:
                result = await self._reward_llm.ainvoke([HumanMessage(content=prompt)])
                match = re.search(r'(\d+(?:\.\d+)?)', result.content)
                if match:
                    return float(match.group(1))
                return 5.0
            except Exception:
                return 5.0

        tasks = [score_one(r) for r in responses]
        scores = await gather_with_concurrency(tasks, max_concurrent=5)

        return [s if isinstance(s, float) else 5.0 for s in scores]


# =============================================================================
# Convenience Functions
# =============================================================================

_tot: Optional[TreeOfThoughts] = None
_bon: Optional[BestOfN] = None
_lock = asyncio.Lock()


async def get_tree_of_thoughts(config: Optional[ToTConfig] = None) -> TreeOfThoughts:
    """Get or create ToT singleton."""
    global _tot
    if _tot is None:
        async with _lock:
            if _tot is None:
                _tot = TreeOfThoughts(config)
    return _tot


async def get_best_of_n(config: Optional[BestOfNConfig] = None) -> BestOfN:
    """Get or create Best-of-N singleton."""
    global _bon
    if _bon is None:
        async with _lock:
            if _bon is None:
                _bon = BestOfN(config)
    return _bon


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    "ToTConfig",
    "BestOfNConfig",
    "SearchStrategy",
    # Results
    "ThoughtNode",
    "ToTResult",
    "BestOfNResult",
    # Classes
    "TreeOfThoughts",
    "BestOfN",
    # Factory
    "get_tree_of_thoughts",
    "get_best_of_n",
]
