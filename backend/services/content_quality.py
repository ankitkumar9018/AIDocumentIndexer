"""
Content Quality Scoring Service
================================

Evaluates and scores generated content quality across multiple dimensions:
- Relevance: How well content matches the topic and source materials
- Readability: Sentence structure, variety, and accessibility
- Coherence: Logical flow, transitions, and structure
- Accuracy: Verification of facts/numbers against sources
- Completeness: Coverage of key topics and depth
- Consistency: Terminology and style consistency across sections
"""

import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from collections import Counter

import structlog

# Try to import langdetect for language consistency checking
try:
    from langdetect import detect, DetectorFactory
    # Make langdetect deterministic
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass
class QualityDimension:
    """Score for a single quality dimension."""
    name: str
    score: float  # 0.0 to 1.0
    weight: float  # Weight in overall score
    feedback: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Complete quality assessment report."""
    overall_score: float
    dimensions: List[QualityDimension]
    summary: str
    needs_revision: bool
    critical_issues: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)


class ContentQualityScorer:
    """
    Evaluates content quality using multiple dimensions.

    Uses heuristic analysis for speed, with optional LLM-based
    deep evaluation for critical content.
    """

    # Quality dimension weights
    DIMENSION_WEIGHTS = {
        "relevance": 0.25,
        "readability": 0.15,
        "coherence": 0.18,
        "accuracy": 0.18,
        "completeness": 0.09,
        "consistency": 0.08,
        "language": 0.07,  # Language consistency
    }

    # Minimum score threshold
    MIN_ACCEPTABLE_SCORE = 0.7

    # Common transition words for coherence analysis
    TRANSITION_WORDS = {
        "addition": ["additionally", "furthermore", "moreover", "also", "besides"],
        "contrast": ["however", "nevertheless", "on the other hand", "conversely", "yet"],
        "cause": ["therefore", "consequently", "thus", "as a result", "hence"],
        "example": ["for example", "for instance", "specifically", "namely", "such as"],
        "conclusion": ["in conclusion", "finally", "to summarize", "in summary", "overall"],
        "sequence": ["first", "second", "next", "then", "finally", "lastly"],
    }

    def __init__(self, min_score: float = 0.7):
        """
        Initialize the quality scorer.

        Args:
            min_score: Minimum acceptable quality score (default 0.7)
        """
        self.min_score = min_score

    async def score_section(
        self,
        content: str,
        title: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        other_sections: Optional[List[str]] = None,
    ) -> QualityReport:
        """
        Score a single section's quality.

        Args:
            content: Section content to evaluate
            title: Section title
            sources: List of source documents/chunks used
            context: Additional context (document title, description, audience, etc.)
            other_sections: Content from other sections for consistency check

        Returns:
            QualityReport with scores and feedback
        """
        dimensions = []

        # Relevance scoring
        relevance = self._score_relevance(content, title, sources, context)
        dimensions.append(relevance)

        # Readability scoring
        readability = self._score_readability(content)
        dimensions.append(readability)

        # Coherence scoring
        coherence = self._score_coherence(content)
        dimensions.append(coherence)

        # Accuracy scoring (basic without LLM)
        accuracy = self._score_accuracy(content, sources)
        dimensions.append(accuracy)

        # Completeness scoring
        completeness = self._score_completeness(content, title, context)
        dimensions.append(completeness)

        # Consistency scoring
        consistency = self._score_consistency(content, other_sections)
        dimensions.append(consistency)

        # Language consistency scoring
        expected_language = context.get('language', 'en') if context else 'en'
        language = self._score_language_consistency(content, expected_language)
        dimensions.append(language)

        # Calculate overall score
        overall_score = sum(
            d.score * d.weight for d in dimensions
        )

        # Generate summary and recommendations
        critical_issues = []
        improvements = []

        for dim in dimensions:
            if dim.score < 0.5:
                critical_issues.extend(dim.feedback)
            elif dim.score < 0.7:
                improvements.extend(dim.suggestions)

        needs_revision = overall_score < self.min_score or len(critical_issues) > 0

        summary = self._generate_summary(overall_score, dimensions, critical_issues)

        return QualityReport(
            overall_score=overall_score,
            dimensions=dimensions,
            summary=summary,
            needs_revision=needs_revision,
            critical_issues=critical_issues,
            improvements=improvements,
        )

    def _score_relevance(
        self,
        content: str,
        title: str,
        sources: Optional[List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]],
    ) -> QualityDimension:
        """Score how relevant the content is to the topic."""
        feedback = []
        suggestions = []
        score = 0.8  # Base score

        content_lower = content.lower()
        title_words = set(title.lower().split())

        # Check if title words appear in content
        title_word_matches = sum(1 for w in title_words if w in content_lower and len(w) > 3)
        title_coverage = title_word_matches / max(len([w for w in title_words if len(w) > 3]), 1)

        if title_coverage < 0.5:
            score -= 0.2
            feedback.append("Content doesn't clearly address the section title")
            suggestions.append("Ensure key terms from the title appear in the content")

        # Check source relevance if sources provided
        if sources:
            source_terms = set()
            for source in sources:
                snippet = source.get('snippet', '') or ''
                source_terms.update(
                    w.lower() for w in snippet.split()
                    if len(w) > 4 and w.isalpha()
                )

            if source_terms:
                content_words = set(w.lower() for w in content.split() if len(w) > 4)
                overlap = len(content_words & source_terms) / max(len(source_terms), 1)
                if overlap < 0.3:
                    score -= 0.15
                    suggestions.append("Consider incorporating more information from source materials")

        # Check context relevance
        if context:
            doc_title = context.get('title', '')
            if doc_title and doc_title.lower() not in content_lower:
                # Not critical, but note it
                suggestions.append("Consider referencing the overall document topic")

        score = max(0.0, min(1.0, score))

        return QualityDimension(
            name="Relevance",
            score=score,
            weight=self.DIMENSION_WEIGHTS["relevance"],
            feedback=feedback,
            suggestions=suggestions,
        )

    def _score_readability(self, content: str) -> QualityDimension:
        """Score readability using sentence length and structure analysis."""
        feedback = []
        suggestions = []
        score = 0.85

        sentences = self._split_sentences(content)
        if not sentences:
            return QualityDimension(
                name="Readability",
                score=0.5,
                weight=self.DIMENSION_WEIGHTS["readability"],
                feedback=["Content appears to have no complete sentences"],
                suggestions=["Ensure content has proper sentence structure"],
            )

        # Analyze sentence lengths
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = statistics.mean(sentence_lengths)
        length_variance = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0

        # Ideal average sentence length: 15-20 words
        if avg_length < 10:
            score -= 0.1
            suggestions.append("Sentences are very short; consider combining for better flow")
        elif avg_length > 30:
            score -= 0.2
            feedback.append("Sentences are too long; break them down for readability")

        # Check for sentence variety
        if length_variance < 3:
            score -= 0.1
            suggestions.append("Vary sentence lengths for better rhythm")

        # Check for very long sentences (>40 words)
        long_sentences = sum(1 for l in sentence_lengths if l > 40)
        if long_sentences > len(sentences) * 0.2:
            score -= 0.15
            feedback.append("Too many excessively long sentences")

        # Check for passive voice (simplified check)
        passive_patterns = [' was ', ' were ', ' been ', ' being ', ' is being ', ' are being ']
        passive_count = sum(content.lower().count(p) for p in passive_patterns)
        passive_ratio = passive_count / max(len(sentences), 1)
        if passive_ratio > 0.4:
            suggestions.append("Consider using more active voice")

        score = max(0.0, min(1.0, score))

        return QualityDimension(
            name="Readability",
            score=score,
            weight=self.DIMENSION_WEIGHTS["readability"],
            feedback=feedback,
            suggestions=suggestions,
        )

    def _score_coherence(self, content: str) -> QualityDimension:
        """Score logical flow and use of transitions."""
        feedback = []
        suggestions = []
        score = 0.8

        content_lower = content.lower()
        sentences = self._split_sentences(content)

        # Count transition words
        transition_count = 0
        transition_types_used = set()
        for category, words in self.TRANSITION_WORDS.items():
            for word in words:
                if word in content_lower:
                    transition_count += 1
                    transition_types_used.add(category)

        # Good coherence has varied transitions
        transitions_per_sentence = transition_count / max(len(sentences), 1)

        if transitions_per_sentence < 0.1 and len(sentences) > 3:
            score -= 0.2
            suggestions.append("Add transition words to improve flow between ideas")
        elif transitions_per_sentence > 0.5:
            suggestions.append("Consider reducing transition word usage for conciseness")

        if len(transition_types_used) < 2 and len(sentences) > 5:
            suggestions.append("Use varied transition types (addition, contrast, conclusion)")

        # Check for paragraph structure
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) == 1 and len(sentences) > 5:
            suggestions.append("Consider breaking content into multiple paragraphs")

        # Check for abrupt topic shifts (simplified)
        if len(sentences) > 3:
            for i in range(1, len(sentences)):
                prev_words = set(sentences[i-1].lower().split())
                curr_words = set(sentences[i].lower().split())
                overlap = len(prev_words & curr_words)
                if overlap == 0:
                    # Potential abrupt shift
                    score -= 0.05

        score = max(0.0, min(1.0, score))

        return QualityDimension(
            name="Coherence",
            score=score,
            weight=self.DIMENSION_WEIGHTS["coherence"],
            feedback=feedback,
            suggestions=suggestions,
        )

    def _score_accuracy(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]],
    ) -> QualityDimension:
        """Score accuracy based on number verification and source consistency."""
        feedback = []
        suggestions = []
        score = 0.85

        # Extract numbers from content
        numbers_in_content = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?%?\b', content)

        if numbers_in_content:
            # If we have sources, check if numbers appear there
            if sources:
                source_text = ' '.join(
                    source.get('snippet', '') or ''
                    for source in sources
                )
                numbers_in_sources = set(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?%?\b', source_text))

                # Check how many content numbers appear in sources
                verified = sum(1 for n in numbers_in_content if n in numbers_in_sources)
                verification_rate = verified / len(numbers_in_content)

                if verification_rate < 0.5:
                    suggestions.append("Some numbers couldn't be verified against sources")
                    score -= 0.1
            else:
                suggestions.append("Numbers present but no sources to verify against")

        # Check for hedging language (indicates uncertainty)
        hedging_words = ['approximately', 'roughly', 'about', 'around', 'estimated', 'possibly', 'might', 'may']
        hedging_count = sum(1 for w in hedging_words if w in content.lower())
        if hedging_count > 3:
            suggestions.append("High use of hedging language; consider being more definitive where appropriate")

        score = max(0.0, min(1.0, score))

        return QualityDimension(
            name="Accuracy",
            score=score,
            weight=self.DIMENSION_WEIGHTS["accuracy"],
            feedback=feedback,
            suggestions=suggestions,
        )

    def _score_completeness(
        self,
        content: str,
        title: str,
        context: Optional[Dict[str, Any]],
    ) -> QualityDimension:
        """Score content completeness and depth."""
        feedback = []
        suggestions = []
        score = 0.8

        word_count = len(content.split())

        # Check minimum word count based on section type
        min_words = 100  # Default minimum
        if context:
            format_type = context.get('output_format', 'docx')
            if format_type == 'pptx':
                min_words = 50  # Slides need less text
            elif format_type in ['md', 'txt']:
                min_words = 75

        if word_count < min_words:
            score -= 0.2
            feedback.append(f"Content is too short ({word_count} words, minimum {min_words})")
            suggestions.append("Expand the content with more details and examples")
        elif word_count < min_words * 1.5:
            suggestions.append("Content meets minimum but could benefit from more depth")

        # Check for examples or specifics
        example_indicators = ['for example', 'for instance', 'such as', 'specifically', 'in particular']
        has_examples = any(ind in content.lower() for ind in example_indicators)
        if not has_examples and word_count > 100:
            suggestions.append("Consider adding concrete examples to illustrate points")

        # Check for structure (bullet points, numbered lists)
        has_structure = bool(re.search(r'(?:^|\n)\s*[•\-\*\d]\.\s', content))
        if not has_structure and word_count > 200:
            suggestions.append("Consider using bullet points or lists for key information")

        score = max(0.0, min(1.0, score))

        return QualityDimension(
            name="Completeness",
            score=score,
            weight=self.DIMENSION_WEIGHTS["completeness"],
            feedback=feedback,
            suggestions=suggestions,
        )

    def _score_consistency(
        self,
        content: str,
        other_sections: Optional[List[str]],
    ) -> QualityDimension:
        """Score terminology and style consistency across sections."""
        feedback = []
        suggestions = []
        score = 0.9  # Start high, deduct for inconsistencies

        if not other_sections:
            return QualityDimension(
                name="Consistency",
                score=score,
                weight=self.DIMENSION_WEIGHTS["consistency"],
                feedback=feedback,
                suggestions=suggestions,
            )

        # Extract key terms from current content
        current_terms = self._extract_key_terms(content)

        # Extract key terms from other sections
        all_other_terms = Counter()
        for section in other_sections:
            all_other_terms.update(self._extract_key_terms(section))

        # Check for term overlap (consistency)
        overlap = len(set(current_terms.keys()) & set(all_other_terms.keys()))
        if overlap < 3 and len(current_terms) > 5:
            suggestions.append("Consider using consistent terminology across sections")
            score -= 0.1

        # Check for style consistency (average sentence length)
        current_avg_len = statistics.mean(
            len(s.split()) for s in self._split_sentences(content)
        ) if self._split_sentences(content) else 0

        other_avg_lens = []
        for section in other_sections:
            sents = self._split_sentences(section)
            if sents:
                other_avg_lens.append(statistics.mean(len(s.split()) for s in sents))

        if other_avg_lens:
            overall_avg = statistics.mean(other_avg_lens)
            if abs(current_avg_len - overall_avg) > 10:
                suggestions.append("Sentence length differs significantly from other sections")
                score -= 0.1

        score = max(0.0, min(1.0, score))

        return QualityDimension(
            name="Consistency",
            score=score,
            weight=self.DIMENSION_WEIGHTS["consistency"],
            feedback=feedback,
            suggestions=suggestions,
        )

    def _score_language_consistency(
        self,
        content: str,
        expected_language: str = 'en',
    ) -> QualityDimension:
        """
        Score language consistency throughout the content.

        Checks if the content maintains the expected language throughout,
        flagging any sentences that appear to be in a different language.
        """
        feedback = []
        suggestions = []
        score = 1.0  # Start at perfect score

        if not LANGDETECT_AVAILABLE:
            # If langdetect isn't available, return neutral score
            return QualityDimension(
                name="Language",
                score=0.85,
                weight=self.DIMENSION_WEIGHTS["language"],
                feedback=[],
                suggestions=["Install langdetect for language consistency checking"],
            )

        sentences = self._split_sentences(content)
        if len(sentences) < 2:
            return QualityDimension(
                name="Language",
                score=score,
                weight=self.DIMENSION_WEIGHTS["language"],
                feedback=feedback,
                suggestions=suggestions,
            )

        foreign_sentences = []
        mixed_language_count = 0

        for sentence in sentences:
            # Only check sentences with enough text for reliable detection
            if len(sentence.strip()) < 30:
                continue

            try:
                detected = detect(sentence)
                if detected != expected_language:
                    mixed_language_count += 1
                    # Store first few foreign sentences for feedback
                    if len(foreign_sentences) < 3:
                        foreign_sentences.append(sentence[:80] + '...' if len(sentence) > 80 else sentence)
            except Exception as e:
                # langdetect can throw exceptions for unusual text
                logger.debug("Language detection failed for sentence", error=str(e))

        # Calculate penalty based on ratio of foreign sentences
        if sentences:
            foreign_ratio = mixed_language_count / len(sentences)

            if foreign_ratio > 0.3:
                score -= 0.4
                feedback.append(f"Significant portion of content ({mixed_language_count} sentences) appears to be in a different language than {expected_language}")
            elif foreign_ratio > 0.1:
                score -= 0.2
                suggestions.append(f"Some content appears to be in a different language than {expected_language}")
            elif mixed_language_count > 0:
                score -= 0.1
                suggestions.append(f"A few sentences may be in a different language")

            if foreign_sentences:
                feedback.append(f"Foreign language detected in: '{foreign_sentences[0][:50]}...'")

        score = max(0.0, min(1.0, score))

        return QualityDimension(
            name="Language",
            score=score,
            weight=self.DIMENSION_WEIGHTS["language"],
            feedback=feedback,
            suggestions=suggestions,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s) > 10]

    def _extract_key_terms(self, text: str) -> Counter:
        """Extract key terms (nouns, technical terms) from text."""
        # Simple extraction: words that are capitalized or longer than 6 chars
        words = re.findall(r'\b[A-Z][a-z]+|\b\w{7,}\b', text)
        return Counter(w.lower() for w in words if not w.isupper())

    def _generate_summary(
        self,
        score: float,
        dimensions: List[QualityDimension],
        critical_issues: List[str],
    ) -> str:
        """Generate a human-readable summary."""
        if score >= 0.9:
            quality = "excellent"
        elif score >= 0.8:
            quality = "good"
        elif score >= 0.7:
            quality = "acceptable"
        elif score >= 0.5:
            quality = "needs improvement"
        else:
            quality = "poor"

        summary = f"Content quality is {quality} (score: {score:.2f})."

        lowest_dim = min(dimensions, key=lambda d: d.score)
        if lowest_dim.score < 0.7:
            summary += f" The weakest area is {lowest_dim.name.lower()}."

        if critical_issues:
            summary += f" {len(critical_issues)} critical issue(s) require attention."

        return summary


# Singleton instance for easy access
quality_scorer = ContentQualityScorer()


async def score_content(
    content: str,
    title: str,
    sources: Optional[List[Dict]] = None,
    context: Optional[Dict] = None,
    other_sections: Optional[List[str]] = None,
) -> QualityReport:
    """Convenience function to score content quality."""
    return await quality_scorer.score_section(
        content, title, sources, context, other_sections
    )
