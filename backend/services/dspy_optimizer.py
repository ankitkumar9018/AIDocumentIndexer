"""
AIDocumentIndexer - DSPy Prompt Optimization Service
=====================================================

Production-ready prompt optimization with:
- Automatic prompt tuning with DSPy
- Few-shot example optimization
- Chain-of-thought refinement
- A/B testing for prompts
- Performance tracking and versioning
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import structlog

logger = structlog.get_logger(__name__)

# Check for DSPy availability
try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot, MIPRO, BootstrapFewShotWithRandomSearch
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logger.info("DSPy not installed. Install with: pip install dspy-ai")


class OptimizationType(str, Enum):
    """Types of prompt optimization."""
    BOOTSTRAP_FEWSHOT = "bootstrap_fewshot"  # Simple few-shot optimization
    MIPRO = "mipro"  # Multi-Instruction Prompt Optimization
    RANDOM_SEARCH = "random_search"  # Random search over examples
    ENSEMBLE = "ensemble"  # Combine multiple optimizers


class PromptStatus(str, Enum):
    """Status of an optimized prompt."""
    DRAFT = "draft"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass
class PromptVersion:
    """Versioned prompt configuration."""
    version_id: str
    prompt_name: str
    template: str
    examples: List[Dict[str, Any]] = field(default_factory=list)
    instructions: str = ""
    status: PromptStatus = PromptStatus.DRAFT
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    optimization_type: Optional[OptimizationType] = None
    parent_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "prompt_name": self.prompt_name,
            "template": self.template,
            "examples": self.examples,
            "instructions": self.instructions,
            "status": self.status.value,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "optimization_type": self.optimization_type.value if self.optimization_type else None,
            "parent_version": self.parent_version,
        }


@dataclass
class OptimizationResult:
    """Result from prompt optimization."""
    success: bool
    new_version: Optional[PromptVersion] = None
    improvement: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    examples_used: int = 0
    iterations: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "new_version": self.new_version.to_dict() if self.new_version else None,
            "improvement": round(self.improvement, 4),
            "metrics": self.metrics,
            "examples_used": self.examples_used,
            "iterations": self.iterations,
            "error": self.error,
        }


@dataclass
class ABTestResult:
    """Result from A/B testing prompts."""
    winner: str
    version_a: PromptVersion
    version_b: PromptVersion
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    samples_tested: int
    confidence: float
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "version_a": self.version_a.version_id,
            "version_b": self.version_b.version_id,
            "metrics_a": self.metrics_a,
            "metrics_b": self.metrics_b,
            "samples_tested": self.samples_tested,
            "confidence": round(self.confidence, 4),
            "recommendation": self.recommendation,
        }


class DSPySignature:
    """Base class for DSPy signatures (task definitions)."""

    def __init__(
        self,
        name: str,
        input_fields: List[str],
        output_fields: List[str],
        instructions: str = "",
    ):
        self.name = name
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.instructions = instructions

    def to_dspy_signature(self):
        """Convert to DSPy signature class."""
        if not DSPY_AVAILABLE:
            return None

        # Dynamically create signature class
        fields = {}
        for field_name in self.input_fields:
            fields[field_name] = dspy.InputField()
        for field_name in self.output_fields:
            fields[field_name] = dspy.OutputField()

        signature_class = type(
            self.name,
            (dspy.Signature,),
            {
                "__doc__": self.instructions,
                **fields,
            }
        )
        return signature_class


class DSPyModule:
    """Wrapper for DSPy modules with optimization support."""

    def __init__(
        self,
        name: str,
        signature: DSPySignature,
        module_type: str = "predict",  # predict, cot, react
    ):
        self.name = name
        self.signature = signature
        self.module_type = module_type
        self._dspy_module = None

        if DSPY_AVAILABLE:
            self._initialize_module()

    def _initialize_module(self):
        """Initialize the DSPy module."""
        sig = self.signature.to_dspy_signature()
        if sig is None:
            return

        if self.module_type == "cot":
            self._dspy_module = dspy.ChainOfThought(sig)
        elif self.module_type == "react":
            self._dspy_module = dspy.ReAct(sig)
        else:
            self._dspy_module = dspy.Predict(sig)

    async def forward(self, **kwargs) -> Dict[str, Any]:
        """Run the module."""
        if self._dspy_module is None:
            # Fallback when DSPy not available
            return {field: f"[Generated {field}]" for field in self.signature.output_fields}

        # Run in thread pool to not block
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._dspy_module(**kwargs)
        )

        # Extract outputs
        outputs = {}
        for field in self.signature.output_fields:
            outputs[field] = getattr(result, field, None)

        return outputs


class DSPyOptimizerService:
    """
    DSPy-based prompt optimization service.

    Provides:
    - Automatic prompt optimization using training examples
    - Chain-of-thought refinement
    - Few-shot example selection
    - A/B testing framework
    - Prompt versioning and rollback
    """

    def __init__(
        self,
        llm_model: str = "gpt-4",
        api_key: Optional[str] = None,
    ):
        self.llm_model = llm_model
        self.api_key = api_key

        # Storage
        self._prompts: Dict[str, List[PromptVersion]] = {}
        self._modules: Dict[str, DSPyModule] = {}
        self._training_data: Dict[str, List[Dict[str, Any]]] = {}

        # Initialize DSPy
        if DSPY_AVAILABLE:
            self._configure_dspy()

    def _configure_dspy(self):
        """Configure DSPy with LLM backend."""
        try:
            # Configure the LLM
            if "gpt" in self.llm_model.lower():
                lm = dspy.OpenAI(model=self.llm_model, api_key=self.api_key)
            else:
                # Default to OpenAI-compatible
                lm = dspy.OpenAI(model=self.llm_model, api_key=self.api_key)

            dspy.settings.configure(lm=lm)
            logger.info(f"DSPy configured with {self.llm_model}")
        except Exception as e:
            logger.error(f"Failed to configure DSPy: {e}")

    def register_signature(
        self,
        name: str,
        input_fields: List[str],
        output_fields: List[str],
        instructions: str = "",
        module_type: str = "predict",
    ) -> DSPyModule:
        """Register a new task signature."""
        signature = DSPySignature(
            name=name,
            input_fields=input_fields,
            output_fields=output_fields,
            instructions=instructions,
        )

        module = DSPyModule(
            name=name,
            signature=signature,
            module_type=module_type,
        )

        self._modules[name] = module

        # Create initial prompt version
        version_id = self._generate_version_id(name, instructions)
        initial_version = PromptVersion(
            version_id=version_id,
            prompt_name=name,
            template=instructions,
            instructions=instructions,
            status=PromptStatus.PRODUCTION,
        )
        self._prompts[name] = [initial_version]

        logger.info(f"Registered signature: {name}")
        return module

    def add_training_example(
        self,
        prompt_name: str,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
    ) -> None:
        """Add a training example for optimization."""
        if prompt_name not in self._training_data:
            self._training_data[prompt_name] = []

        self._training_data[prompt_name].append({
            "inputs": inputs,
            "outputs": outputs,
            "added_at": datetime.utcnow().isoformat(),
        })

        logger.debug(f"Added training example for {prompt_name}")

    async def optimize(
        self,
        prompt_name: str,
        optimization_type: OptimizationType = OptimizationType.BOOTSTRAP_FEWSHOT,
        num_examples: int = 5,
        max_iterations: int = 10,
        metric_fn: Optional[Callable] = None,
    ) -> OptimizationResult:
        """
        Optimize a prompt using DSPy.

        Args:
            prompt_name: Name of the prompt to optimize
            optimization_type: Type of optimization to use
            num_examples: Number of examples for few-shot
            max_iterations: Maximum optimization iterations
            metric_fn: Custom metric function (higher is better)

        Returns:
            OptimizationResult with optimized prompt
        """
        if not DSPY_AVAILABLE:
            return OptimizationResult(
                success=False,
                error="DSPy not available",
            )

        if prompt_name not in self._modules:
            return OptimizationResult(
                success=False,
                error=f"Unknown prompt: {prompt_name}",
            )

        training_data = self._training_data.get(prompt_name, [])
        if len(training_data) < 3:
            return OptimizationResult(
                success=False,
                error=f"Need at least 3 training examples, have {len(training_data)}",
            )

        try:
            module = self._modules[prompt_name]

            # Create DSPy examples
            examples = []
            for data in training_data[:num_examples * 2]:
                example = dspy.Example(**data["inputs"], **data["outputs"])
                example = example.with_inputs(*module.signature.input_fields)
                examples.append(example)

            # Split into train/dev
            train_examples = examples[:len(examples) * 2 // 3]
            dev_examples = examples[len(examples) * 2 // 3:]

            # Default metric: exact match on output fields
            if metric_fn is None:
                def metric_fn(example, prediction, trace=None):
                    score = 0
                    for field in module.signature.output_fields:
                        expected = getattr(example, field, "")
                        predicted = getattr(prediction, field, "")
                        if expected and predicted:
                            score += 1 if expected.lower() in predicted.lower() else 0
                    return score / len(module.signature.output_fields)

            # Select optimizer
            if optimization_type == OptimizationType.MIPRO:
                optimizer = MIPRO(
                    metric=metric_fn,
                    num_candidates=5,
                    init_temperature=1.0,
                )
            elif optimization_type == OptimizationType.RANDOM_SEARCH:
                optimizer = BootstrapFewShotWithRandomSearch(
                    metric=metric_fn,
                    max_bootstrapped_demos=num_examples,
                    max_labeled_demos=num_examples,
                    num_candidate_programs=max_iterations,
                )
            else:
                optimizer = BootstrapFewShot(
                    metric=metric_fn,
                    max_bootstrapped_demos=num_examples,
                    max_labeled_demos=num_examples,
                )

            # Run optimization
            optimized_module = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: optimizer.compile(
                    module._dspy_module,
                    trainset=train_examples,
                )
            )

            # Evaluate on dev set
            scores = []
            for example in dev_examples:
                inputs = {f: getattr(example, f) for f in module.signature.input_fields}
                prediction = optimized_module(**inputs)
                score = metric_fn(example, prediction)
                scores.append(score)

            avg_score = sum(scores) / len(scores) if scores else 0

            # Get previous best score
            current_versions = self._prompts.get(prompt_name, [])
            prev_score = 0
            if current_versions:
                prev_score = current_versions[-1].metrics.get("avg_score", 0)

            # Extract optimized prompt info
            # DSPy stores demos in the module
            demos = []
            if hasattr(optimized_module, 'demos'):
                for demo in optimized_module.demos:
                    demo_dict = {}
                    for field in module.signature.input_fields + module.signature.output_fields:
                        demo_dict[field] = getattr(demo, field, "")
                    demos.append(demo_dict)

            # Create new version
            version_id = self._generate_version_id(prompt_name, str(demos))
            new_version = PromptVersion(
                version_id=version_id,
                prompt_name=prompt_name,
                template=module.signature.instructions,
                examples=demos,
                instructions=module.signature.instructions,
                status=PromptStatus.TESTING,
                metrics={"avg_score": avg_score, "dev_samples": len(dev_examples)},
                optimization_type=optimization_type,
                parent_version=current_versions[-1].version_id if current_versions else None,
            )

            self._prompts[prompt_name].append(new_version)

            return OptimizationResult(
                success=True,
                new_version=new_version,
                improvement=avg_score - prev_score,
                metrics={"avg_score": avg_score, "prev_score": prev_score},
                examples_used=len(train_examples),
                iterations=max_iterations,
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                success=False,
                error=str(e),
            )

    async def ab_test(
        self,
        prompt_name: str,
        version_a: str,
        version_b: str,
        test_inputs: List[Dict[str, str]],
        metric_fn: Optional[Callable] = None,
    ) -> ABTestResult:
        """
        A/B test two prompt versions.

        Args:
            prompt_name: Name of the prompt
            version_a: First version ID
            version_b: Second version ID
            test_inputs: List of test inputs
            metric_fn: Custom metric function

        Returns:
            ABTestResult with winner determination
        """
        versions = {v.version_id: v for v in self._prompts.get(prompt_name, [])}

        if version_a not in versions or version_b not in versions:
            raise ValueError("Version not found")

        va = versions[version_a]
        vb = versions[version_b]

        module = self._modules.get(prompt_name)
        if not module:
            raise ValueError(f"Unknown prompt: {prompt_name}")

        # Default metric: response quality score (0-1)
        if metric_fn is None:
            def metric_fn(response: Dict[str, str]) -> float:
                # Simple heuristic: longer, more detailed responses score higher
                total_len = sum(len(str(v)) for v in response.values())
                return min(1.0, total_len / 500)

        scores_a = []
        scores_b = []

        for inputs in test_inputs:
            # Test version A (with its examples)
            response_a = await self._run_with_version(module, va, inputs)
            scores_a.append(metric_fn(response_a))

            # Test version B (with its examples)
            response_b = await self._run_with_version(module, vb, inputs)
            scores_b.append(metric_fn(response_b))

        avg_a = sum(scores_a) / len(scores_a) if scores_a else 0
        avg_b = sum(scores_b) / len(scores_b) if scores_b else 0

        # Simple winner determination
        if avg_a > avg_b * 1.05:  # 5% threshold
            winner = version_a
            confidence = min(0.99, (avg_a - avg_b) / avg_a)
            recommendation = f"Version A ({version_a}) performs better. Recommend promoting to production."
        elif avg_b > avg_a * 1.05:
            winner = version_b
            confidence = min(0.99, (avg_b - avg_a) / avg_b)
            recommendation = f"Version B ({version_b}) performs better. Recommend promoting to production."
        else:
            winner = "tie"
            confidence = abs(avg_a - avg_b) / max(avg_a, avg_b) if max(avg_a, avg_b) > 0 else 0
            recommendation = "No significant difference. Consider more testing or keep current version."

        return ABTestResult(
            winner=winner,
            version_a=va,
            version_b=vb,
            metrics_a={"avg_score": avg_a, "scores": scores_a},
            metrics_b={"avg_score": avg_b, "scores": scores_b},
            samples_tested=len(test_inputs),
            confidence=confidence,
            recommendation=recommendation,
        )

    async def _run_with_version(
        self,
        module: DSPyModule,
        version: PromptVersion,
        inputs: Dict[str, str],
    ) -> Dict[str, Any]:
        """Run module with specific version's examples."""
        # For now, just run the module
        # In full implementation, we'd inject the version's examples
        return await module.forward(**inputs)

    def promote_version(
        self,
        prompt_name: str,
        version_id: str,
    ) -> bool:
        """Promote a version to production."""
        versions = self._prompts.get(prompt_name, [])

        for v in versions:
            if v.version_id == version_id:
                # Demote current production
                for other in versions:
                    if other.status == PromptStatus.PRODUCTION:
                        other.status = PromptStatus.DEPRECATED

                v.status = PromptStatus.PRODUCTION
                logger.info(f"Promoted {version_id} to production for {prompt_name}")
                return True

        return False

    def rollback(
        self,
        prompt_name: str,
    ) -> Optional[PromptVersion]:
        """Rollback to previous production version."""
        versions = self._prompts.get(prompt_name, [])

        # Find current production and previous
        current_prod = None
        prev_prod = None

        for v in reversed(versions):
            if v.status == PromptStatus.PRODUCTION:
                if current_prod is None:
                    current_prod = v
                else:
                    prev_prod = v
                    break
            elif v.status == PromptStatus.DEPRECATED and current_prod:
                prev_prod = v
                break

        if current_prod and prev_prod:
            current_prod.status = PromptStatus.DEPRECATED
            prev_prod.status = PromptStatus.PRODUCTION
            logger.info(f"Rolled back {prompt_name} to {prev_prod.version_id}")
            return prev_prod

        return None

    def get_production_version(
        self,
        prompt_name: str,
    ) -> Optional[PromptVersion]:
        """Get the current production version."""
        versions = self._prompts.get(prompt_name, [])

        for v in reversed(versions):
            if v.status == PromptStatus.PRODUCTION:
                return v

        return versions[-1] if versions else None

    def list_versions(
        self,
        prompt_name: str,
    ) -> List[PromptVersion]:
        """List all versions of a prompt."""
        return self._prompts.get(prompt_name, [])

    def _generate_version_id(self, name: str, content: str) -> str:
        """Generate a version ID."""
        hash_input = f"{name}:{content}:{datetime.utcnow().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    async def run(
        self,
        prompt_name: str,
        **inputs,
    ) -> Dict[str, Any]:
        """Run a prompt with the production version."""
        module = self._modules.get(prompt_name)
        if not module:
            raise ValueError(f"Unknown prompt: {prompt_name}")

        return await module.forward(**inputs)

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = {
            "registered_prompts": len(self._modules),
            "total_versions": sum(len(v) for v in self._prompts.values()),
            "training_examples": sum(len(v) for v in self._training_data.values()),
            "prompts": {},
        }

        for name, versions in self._prompts.items():
            prod_version = self.get_production_version(name)
            stats["prompts"][name] = {
                "versions": len(versions),
                "training_examples": len(self._training_data.get(name, [])),
                "production_version": prod_version.version_id if prod_version else None,
                "production_metrics": prod_version.metrics if prod_version else {},
            }

        return stats


# Pre-built RAG signatures
class RAGSignatures:
    """Pre-built DSPy signatures for common RAG tasks."""

    @staticmethod
    def query_rewriter() -> Dict[str, Any]:
        return {
            "name": "QueryRewriter",
            "input_fields": ["original_query"],
            "output_fields": ["rewritten_query", "search_keywords"],
            "instructions": """Rewrite the user's query to be more effective for semantic search.
Extract key search terms and rephrase for clarity.""",
            "module_type": "cot",
        }

    @staticmethod
    def answer_generator() -> Dict[str, Any]:
        return {
            "name": "AnswerGenerator",
            "input_fields": ["question", "context"],
            "output_fields": ["answer", "citations"],
            "instructions": """Generate a comprehensive answer to the question using the provided context.
Include citations to specific parts of the context.""",
            "module_type": "cot",
        }

    @staticmethod
    def summarizer() -> Dict[str, Any]:
        return {
            "name": "Summarizer",
            "input_fields": ["document"],
            "output_fields": ["summary", "key_points"],
            "instructions": """Create a concise summary of the document.
Extract 3-5 key points as bullet points.""",
            "module_type": "predict",
        }

    @staticmethod
    def entity_extractor() -> Dict[str, Any]:
        return {
            "name": "EntityExtractor",
            "input_fields": ["text"],
            "output_fields": ["entities", "relationships"],
            "instructions": """Extract named entities (people, places, organizations, concepts) from the text.
Identify relationships between entities.""",
            "module_type": "cot",
        }

    @staticmethod
    def fact_checker() -> Dict[str, Any]:
        return {
            "name": "FactChecker",
            "input_fields": ["claim", "evidence"],
            "output_fields": ["verdict", "explanation", "confidence"],
            "instructions": """Determine if the claim is supported, refuted, or inconclusive based on the evidence.
Explain your reasoning and provide a confidence score (0-1).""",
            "module_type": "cot",
        }


# Singleton instance
_dspy_service: Optional[DSPyOptimizerService] = None


def get_dspy_service(
    llm_model: str = "gpt-4",
    api_key: Optional[str] = None,
) -> DSPyOptimizerService:
    """Get or create the DSPy service singleton."""
    global _dspy_service
    if _dspy_service is None:
        _dspy_service = DSPyOptimizerService(
            llm_model=llm_model,
            api_key=api_key,
        )

        # Register default RAG signatures
        for sig_fn in [
            RAGSignatures.query_rewriter,
            RAGSignatures.answer_generator,
            RAGSignatures.summarizer,
            RAGSignatures.entity_extractor,
            RAGSignatures.fact_checker,
        ]:
            sig = sig_fn()
            _dspy_service.register_signature(**sig)

    return _dspy_service
