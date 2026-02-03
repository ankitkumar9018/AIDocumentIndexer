"""
AIDocumentIndexer - Smart Highlights Service
=============================================

AI-powered reading mode with intelligent highlighting:
- Key point extraction
- Important entity highlighting
- Summary generation
- Reading time estimation
- Difficulty level assessment
- Interactive annotations
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib

import structlog

logger = structlog.get_logger(__name__)


class HighlightType(str, Enum):
    """Types of highlights."""
    KEY_POINT = "key_point"  # Important statements
    ENTITY = "entity"  # People, places, organizations
    DEFINITION = "definition"  # Term definitions
    STATISTIC = "statistic"  # Numbers and statistics
    QUOTE = "quote"  # Notable quotes
    ACTION_ITEM = "action_item"  # Tasks or actions
    WARNING = "warning"  # Warnings or caveats
    QUESTION = "question"  # Questions raised
    CONCLUSION = "conclusion"  # Concluding statements
    EXAMPLE = "example"  # Examples and illustrations


class DifficultyLevel(str, Enum):
    """Reading difficulty levels."""
    EASY = "easy"  # Simple language, short sentences
    MEDIUM = "medium"  # Standard complexity
    HARD = "hard"  # Technical or complex content
    EXPERT = "expert"  # Highly specialized content


@dataclass
class Highlight:
    """Represents a highlighted section."""
    id: str
    highlight_type: HighlightType
    text: str
    start_offset: int
    end_offset: int
    confidence: float
    explanation: Optional[str] = None
    related_highlights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.highlight_type.value,
            "text": self.text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "related_highlights": self.related_highlights,
            "metadata": self.metadata,
        }


@dataclass
class DocumentAnalysis:
    """Complete analysis of a document for reading mode."""
    document_id: str
    title: str
    highlights: List[Highlight]
    summary: str
    key_takeaways: List[str]
    reading_time_minutes: int
    difficulty_level: DifficultyLevel
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    vocabulary_richness: float
    topics: List[str]
    entities: Dict[str, List[str]]
    questions_answered: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "highlights": [h.to_dict() for h in self.highlights],
            "summary": self.summary,
            "key_takeaways": self.key_takeaways,
            "reading_time_minutes": self.reading_time_minutes,
            "difficulty_level": self.difficulty_level.value,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "avg_sentence_length": self.avg_sentence_length,
            "vocabulary_richness": self.vocabulary_richness,
            "topics": self.topics,
            "entities": self.entities,
            "questions_answered": self.questions_answered,
            "created_at": self.created_at.isoformat(),
        }


class SmartHighlightsService:
    """
    Service for intelligent document highlighting and reading mode.

    Features:
    - Automatic key point extraction
    - Entity recognition and highlighting
    - Summary and takeaway generation
    - Reading metrics calculation
    - LLM-powered analysis
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        words_per_minute: int = 200,
        max_highlights: int = 50,
    ):
        self.llm_client = llm_client
        self.words_per_minute = words_per_minute
        self.max_highlights = max_highlights
        self._analysis_cache: Dict[str, DocumentAnalysis] = {}

    async def analyze_document(
        self,
        document_id: str,
        content: str,
        title: Optional[str] = None,
        use_llm: bool = True,
    ) -> DocumentAnalysis:
        """
        Analyze a document and generate highlights.

        Args:
            document_id: Document identifier
            content: Document text content
            title: Optional document title
            use_llm: Whether to use LLM for enhanced analysis

        Returns:
            DocumentAnalysis with highlights and metrics
        """
        # Check cache
        cache_key = hashlib.md5(content.encode()).hexdigest()[:16]
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        # Basic text analysis
        words = content.split()
        sentences = self._split_sentences(content)

        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(sentence_count, 1)

        # Calculate vocabulary richness (type-token ratio)
        unique_words = set(w.lower() for w in words if w.isalpha())
        vocabulary_richness = len(unique_words) / max(word_count, 1)

        # Estimate reading time
        reading_time = max(1, word_count // self.words_per_minute)

        # Assess difficulty
        difficulty = self._assess_difficulty(
            avg_sentence_length,
            vocabulary_richness,
            content
        )

        # Extract highlights
        highlights = await self._extract_highlights(content, use_llm)

        # Extract entities
        entities = self._extract_entities(content)

        # Generate summary and takeaways
        if use_llm and self.llm_client:
            summary, takeaways = await self._generate_summary_with_llm(content)
            topics = await self._extract_topics_with_llm(content)
            questions = await self._extract_questions_with_llm(content)
        else:
            summary = self._generate_extractive_summary(content, sentences)
            takeaways = [h.text for h in highlights if h.highlight_type == HighlightType.KEY_POINT][:5]
            topics = self._extract_topics_basic(content)
            questions = []

        analysis = DocumentAnalysis(
            document_id=document_id,
            title=title or "Untitled",
            highlights=highlights,
            summary=summary,
            key_takeaways=takeaways,
            reading_time_minutes=reading_time,
            difficulty_level=difficulty,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            vocabulary_richness=vocabulary_richness,
            topics=topics,
            entities=entities,
            questions_answered=questions,
        )

        # Cache result
        self._analysis_cache[cache_key] = analysis

        return analysis

    async def _extract_highlights(
        self,
        content: str,
        use_llm: bool = True,
    ) -> List[Highlight]:
        """Extract highlights from content."""
        highlights: List[Highlight] = []

        # Rule-based extraction
        highlights.extend(self._extract_definitions(content))
        highlights.extend(self._extract_statistics(content))
        highlights.extend(self._extract_quotes(content))
        highlights.extend(self._extract_action_items(content))
        highlights.extend(self._extract_questions(content))

        # LLM-based key point extraction
        if use_llm and self.llm_client:
            llm_highlights = await self._extract_key_points_with_llm(content)
            highlights.extend(llm_highlights)
        else:
            # Fallback to heuristic key point extraction
            highlights.extend(self._extract_key_points_heuristic(content))

        # Sort by position and limit
        highlights.sort(key=lambda h: h.start_offset)
        return highlights[:self.max_highlights]

    def _extract_definitions(self, content: str) -> List[Highlight]:
        """Extract term definitions."""
        highlights = []
        patterns = [
            r'([A-Z][a-zA-Z]+)\s+(?:is|are|refers to|means|defined as)\s+([^.]+\.)',
            r'([A-Z][a-zA-Z]+)\s*:\s*([^.]+\.)',
            r'\"([^\"]+)\"\s+(?:is|means)\s+([^.]+\.)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content):
                highlights.append(Highlight(
                    id=self._generate_id("def", match.start()),
                    highlight_type=HighlightType.DEFINITION,
                    text=match.group(0),
                    start_offset=match.start(),
                    end_offset=match.end(),
                    confidence=0.8,
                    explanation=f"Definition of '{match.group(1)}'",
                ))

        return highlights

    def _extract_statistics(self, content: str) -> List[Highlight]:
        """Extract statistics and numbers."""
        highlights = []
        patterns = [
            r'\d+(?:\.\d+)?%\s+[^.]+\.',  # Percentages
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?\s+[^.]+\.',  # Currency
            r'\d+(?:,\d{3})*\s+(?:users|customers|people|companies|employees)[^.]*\.',  # Counts
            r'(?:increased|decreased|grew|fell)\s+(?:by\s+)?\d+(?:\.\d+)?%',  # Changes
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                highlights.append(Highlight(
                    id=self._generate_id("stat", match.start()),
                    highlight_type=HighlightType.STATISTIC,
                    text=match.group(0),
                    start_offset=match.start(),
                    end_offset=match.end(),
                    confidence=0.9,
                ))

        return highlights

    def _extract_quotes(self, content: str) -> List[Highlight]:
        """Extract notable quotes."""
        highlights = []
        pattern = r'["\u201c]([^"\u201d]{20,200})["\u201d]'

        for match in re.finditer(pattern, content):
            highlights.append(Highlight(
                id=self._generate_id("quote", match.start()),
                highlight_type=HighlightType.QUOTE,
                text=match.group(0),
                start_offset=match.start(),
                end_offset=match.end(),
                confidence=0.85,
            ))

        return highlights

    def _extract_action_items(self, content: str) -> List[Highlight]:
        """Extract action items and tasks."""
        highlights = []
        patterns = [
            r'(?:TODO|FIXME|ACTION|TASK):\s*([^\n]+)',
            r'(?:should|must|need to|have to)\s+([^.]+\.)',
            r'\[\s*\]\s*([^\n]+)',  # Checkbox items
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                highlights.append(Highlight(
                    id=self._generate_id("action", match.start()),
                    highlight_type=HighlightType.ACTION_ITEM,
                    text=match.group(0),
                    start_offset=match.start(),
                    end_offset=match.end(),
                    confidence=0.75,
                ))

        return highlights

    def _extract_questions(self, content: str) -> List[Highlight]:
        """Extract questions from content."""
        highlights = []
        pattern = r'[^.!?]*\?'

        for match in re.finditer(pattern, content):
            text = match.group(0).strip()
            if len(text) > 10:  # Skip very short questions
                highlights.append(Highlight(
                    id=self._generate_id("question", match.start()),
                    highlight_type=HighlightType.QUESTION,
                    text=text,
                    start_offset=match.start(),
                    end_offset=match.end(),
                    confidence=0.9,
                ))

        return highlights

    def _extract_key_points_heuristic(self, content: str) -> List[Highlight]:
        """Extract key points using heuristics."""
        highlights = []
        sentences = self._split_sentences(content)

        # Heuristics for important sentences
        importance_indicators = [
            'important', 'key', 'critical', 'essential', 'significant',
            'notably', 'importantly', 'crucially', 'primarily', 'mainly',
            'in conclusion', 'in summary', 'therefore', 'thus', 'hence',
            'the main', 'the key', 'the most', 'first', 'second', 'finally',
        ]

        for sentence in sentences:
            lower_sentence = sentence.lower()
            for indicator in importance_indicators:
                if indicator in lower_sentence:
                    start = content.find(sentence)
                    if start >= 0:
                        highlights.append(Highlight(
                            id=self._generate_id("key", start),
                            highlight_type=HighlightType.KEY_POINT,
                            text=sentence,
                            start_offset=start,
                            end_offset=start + len(sentence),
                            confidence=0.7,
                            explanation=f"Contains importance indicator: '{indicator}'",
                        ))
                    break

        return highlights

    async def _extract_key_points_with_llm(self, content: str) -> List[Highlight]:
        """Extract key points using LLM."""
        if not self.llm_client:
            return []

        try:
            prompt = f"""Extract the 5 most important key points from this text.
For each key point, provide the exact text from the document.

Text:
{content[:3000]}

Respond in JSON format:
{{"key_points": ["exact text 1", "exact text 2", ...]}}"""

            response = await self.llm_client.complete(prompt)
            import json
            result = json.loads(response)

            highlights = []
            for point in result.get("key_points", []):
                start = content.find(point)
                if start >= 0:
                    highlights.append(Highlight(
                        id=self._generate_id("key_llm", start),
                        highlight_type=HighlightType.KEY_POINT,
                        text=point,
                        start_offset=start,
                        end_offset=start + len(point),
                        confidence=0.9,
                        explanation="Identified by AI as key point",
                    ))

            return highlights

        except Exception as e:
            logger.warning("LLM key point extraction failed", error=str(e))
            return []

    async def _generate_summary_with_llm(
        self,
        content: str,
    ) -> Tuple[str, List[str]]:
        """Generate summary and takeaways using LLM."""
        if not self.llm_client:
            return "", []

        try:
            prompt = f"""Analyze this document and provide:
1. A concise summary (2-3 sentences)
2. 3-5 key takeaways (bullet points)

Document:
{content[:4000]}

Respond in JSON format:
{{"summary": "...", "takeaways": ["...", "..."]}}"""

            response = await self.llm_client.complete(prompt)
            import json
            result = json.loads(response)

            return result.get("summary", ""), result.get("takeaways", [])

        except Exception as e:
            logger.warning("LLM summary generation failed", error=str(e))
            return "", []

    async def _extract_topics_with_llm(self, content: str) -> List[str]:
        """Extract main topics using LLM."""
        if not self.llm_client:
            return self._extract_topics_basic(content)

        try:
            prompt = f"""What are the main topics covered in this document?
List 3-5 topics as short phrases.

Document:
{content[:2000]}

Respond in JSON format:
{{"topics": ["topic1", "topic2", ...]}}"""

            response = await self.llm_client.complete(prompt)
            import json
            result = json.loads(response)
            return result.get("topics", [])

        except Exception as e:
            logger.warning("LLM topic extraction failed", error=str(e))
            return self._extract_topics_basic(content)

    async def _extract_questions_with_llm(self, content: str) -> List[str]:
        """Extract questions the document answers using LLM."""
        if not self.llm_client:
            return []

        try:
            prompt = f"""What questions does this document answer?
List 3-5 questions that someone might have that this document addresses.

Document:
{content[:2000]}

Respond in JSON format:
{{"questions": ["question1?", "question2?", ...]}}"""

            response = await self.llm_client.complete(prompt)
            import json
            result = json.loads(response)
            return result.get("questions", [])

        except Exception as e:
            logger.warning("LLM question extraction failed", error=str(e))
            return []

    def _generate_extractive_summary(
        self,
        content: str,
        sentences: List[str],
    ) -> str:
        """Generate extractive summary by selecting important sentences."""
        if not sentences:
            return ""

        # Score sentences
        scored = []
        for sentence in sentences:
            score = 0
            # Position score (early sentences often more important)
            pos = content.find(sentence)
            if pos < len(content) * 0.2:  # First 20%
                score += 2
            # Length score (not too short, not too long)
            words = len(sentence.split())
            if 10 <= words <= 30:
                score += 1
            # Contains important words
            important_words = ['important', 'key', 'main', 'significant', 'conclusion']
            if any(w in sentence.lower() for w in important_words):
                score += 2

            scored.append((sentence, score))

        # Select top sentences
        scored.sort(key=lambda x: x[1], reverse=True)
        summary_sentences = [s[0] for s in scored[:3]]

        # Reorder by position in document
        summary_sentences.sort(key=lambda s: content.find(s))

        return ' '.join(summary_sentences)

    def _extract_topics_basic(self, content: str) -> List[str]:
        """Extract topics using basic keyword extraction."""
        from collections import Counter

        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        word_counts = Counter(words)

        # Filter common words and return top topics
        stop_topics = {'The', 'This', 'That', 'These', 'Those', 'It', 'They'}
        topics = [
            word for word, count in word_counts.most_common(10)
            if word not in stop_topics and count > 1
        ]

        return topics[:5]

    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from content."""
        entities: Dict[str, List[str]] = {
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": [],
        }

        # Simple pattern-based extraction
        # People (names with titles)
        people_pattern = r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        entities["people"] = list(set(re.findall(people_pattern, content)))

        # Organizations (Inc., Corp., LLC, etc.)
        org_pattern = r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:Inc\.|Corp\.|LLC|Ltd\.|Company)'
        entities["organizations"] = list(set(re.findall(org_pattern, content)))

        # Dates
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        entities["dates"] = list(set(re.findall(date_pattern, content)))

        return entities

    def _assess_difficulty(
        self,
        avg_sentence_length: float,
        vocabulary_richness: float,
        content: str,
    ) -> DifficultyLevel:
        """Assess reading difficulty level."""
        # Flesch-Kincaid inspired assessment
        score = 0

        # Sentence length factor
        if avg_sentence_length > 25:
            score += 2
        elif avg_sentence_length > 15:
            score += 1

        # Vocabulary richness factor
        if vocabulary_richness > 0.7:
            score += 2
        elif vocabulary_richness > 0.5:
            score += 1

        # Technical term density
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+(?:tion|ment|ness|ity)\b',  # Complex suffixes
        ]
        tech_count = sum(len(re.findall(p, content)) for p in technical_patterns)
        tech_density = tech_count / max(len(content.split()), 1)

        if tech_density > 0.1:
            score += 2
        elif tech_density > 0.05:
            score += 1

        # Map score to difficulty
        if score >= 5:
            return DifficultyLevel.EXPERT
        elif score >= 3:
            return DifficultyLevel.HARD
        elif score >= 1:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.EASY

    def _split_sentences(self, content: str) -> List[str]:
        """Split content into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        return [s.strip() for s in sentences if s.strip()]

    def _generate_id(self, prefix: str, offset: int) -> str:
        """Generate highlight ID."""
        return f"{prefix}_{offset}"


# Singleton instance
_smart_highlights_service: Optional[SmartHighlightsService] = None


def get_smart_highlights_service() -> SmartHighlightsService:
    """Get or create the smart highlights service singleton."""
    global _smart_highlights_service
    if _smart_highlights_service is None:
        _smart_highlights_service = SmartHighlightsService()
    return _smart_highlights_service
